import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.core.config import get_settings
from src.models.session import ModoSesion, SesionStatus
from src.models.user import Usuario
from src.repositories.session import SessionRepository
from src.repositories.result import ResultRepository
from src.services.storage import StorageService
from src.schemas.result import ResultadoResponse, DetalleResultadoResponse
from src.schemas.session import SesionResponse, SesionDetalleResponse

settings = get_settings()

# Modos válidos por endpoint
_MODOS_IMAGEN = {ModoSesion.IMAGEN_SUBIDA, ModoSesion.FOTO_CAPTURADA}
_MODOS_VIDEO = {ModoSesion.VIDEO_SUBIDO, ModoSesion.VIDEO_GRABADO}

# Singleton del predictor — se carga una vez al importar el módulo
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from ml.inference import ASLPredictor
        _predictor = ASLPredictor(
            model_path=settings.MODEL_PATH,
            task_path=settings.TASK_PATH,
        )
    return _predictor


class PredictionsService:

    def __init__(self, db, usuario: Usuario):
        self.db = db
        self.usuario = usuario
        self.session_repo = SessionRepository(db)
        self.result_repo = ResultRepository(db)
        self.storage = StorageService()

    async def predict_image(
        self,
        archivo: UploadFile,
        modo: ModoSesion,
    ) -> dict:
        # Validar modo
        if modo not in _MODOS_IMAGEN:
            raise HTTPException(
                status_code=422,
                detail=f"Modo inválido para este endpoint. Usa: {[m.value for m in _MODOS_IMAGEN]}"
            )

        # Leer contenido del archivo
        contenido = await archivo.read()

        # Validar que es una imagen decodificable
        imagen_np = self._decodificar_imagen(contenido)

        # Crear sesión en BD
        sesion = await self.session_repo.create_sesion(
            usuario_id=self.usuario.id,
            modo=modo,
            status=SesionStatus.COMPLETADA,
        )

        # Subir a Storage
        try:
            ruta_storage = await self.storage.upload_file(
                usuario_id=self.usuario.id,
                sesion_id=sesion.id,
                nombre_original=archivo.filename or "imagen.jpg",
                contenido=contenido,
                content_type=archivo.content_type or "image/jpeg",
            )
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error al subir el archivo: {str(e)}"
            )

        # Registrar archivo en BD
        await self.session_repo.create_archivo(
            sesion_id=sesion.id,
            nombre_original=archivo.filename or "imagen.jpg",
            ruta_storage=ruta_storage,
            formato=archivo.content_type or "image/jpeg",
            tamano=len(contenido),
        )

        # Ejecutar ML
        try:
            predictor = get_predictor()
            gesto, confianza = predictor.predict(imagen_np)
        except Exception as e:
            await self.session_repo.update_status(sesion, SesionStatus.INTERRUMPIDA)
            await self.db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Error en el modelo ML: {str(e)}"
            )

        # Guardar resultado
        resultado = await self.result_repo.create_resultado(
            sesion_id=sesion.id,
            secuencia_texto=gesto if gesto != "nothing" else "",
            confianza_media=confianza,
            total_frames=1,
        )

        detalles = []
        if gesto != "nothing":
            detalles = await self.result_repo.create_detalles(
                resultado_id=resultado.id,
                detalles=[{
                    "gesto": gesto,
                    "confianza": confianza,
                    "posicion_secuencia": 0,
                    "timestamp_frame": None,
                }]
            )

        await self.db.commit()

        return {
            "sesion": SesionResponse.model_validate(sesion),
            "resultado": ResultadoResponse(
                id=resultado.id,
                sesion_id=resultado.sesion_id,
                secuencia_texto=resultado.secuencia_texto,
                confianza_media=resultado.confianza_media,
                total_frames=resultado.total_frames,
                detalles=[
                    DetalleResultadoResponse.model_validate(d)
                    for d in detalles
                ],
            ),
        }

    async def predict_video(
        self,
        archivo: UploadFile,
        modo: ModoSesion,
    ) -> dict:
        # Validar modo
        if modo not in _MODOS_VIDEO:
            raise HTTPException(
                status_code=422,
                detail=f"Modo inválido para este endpoint. Usa: {[m.value for m in _MODOS_VIDEO]}"
            )

        # Leer contenido del archivo
        contenido = await archivo.read()

        # Validar vídeo y extraer frames — 422 si corrupto o supera duración máxima
        # Ocurre antes de crear sesión en BD: sin efectos secundarios si falla
        from src.services.video import VideoService
        frames = VideoService.validate_and_extract_frames(
            contenido=contenido,
            nombre_original=archivo.filename or "video.mp4",
        )

        # Crear sesión en BD
        sesion = await self.session_repo.create_sesion(
            usuario_id=self.usuario.id,
            modo=modo,
            status=SesionStatus.COMPLETADA,
        )

        # Ejecutar ML sobre cada frame
        try:
            predictor = get_predictor()
            predicciones = []
            for frame in frames:
                gesto, confianza = predictor.predict(frame)
                predicciones.append((gesto, confianza))
        except Exception as e:
            await self.session_repo.update_status(sesion, SesionStatus.INTERRUMPIDA)
            await self.db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Error en el modelo ML: {str(e)}"
            )

        # Deduplicación de consecutivos y construcción de secuencia
        # nothing resetea el último gesto — permite detectar A→nothing→A como dos gestos
        detalles_data = []
        ultimo_gesto = None
        posicion = 0

        for frame_idx, (gesto, confianza) in enumerate(predicciones):
            if gesto == "nothing":
                ultimo_gesto = None
                continue
            if gesto == ultimo_gesto:
                continue
            ultimo_gesto = gesto
            detalles_data.append({
                "gesto": gesto,
                "confianza": confianza,
                "posicion_secuencia": posicion,
                "timestamp_frame": round(frame_idx / settings.VIDEO_FPS_SAMPLE, 2),
            })
            posicion += 1

        secuencia_texto = "".join(d["gesto"] for d in detalles_data)
        confianza_media = (
            sum(d["confianza"] for d in detalles_data) / len(detalles_data)
            if detalles_data else 0.0
        )

        # Guardar resultado
        resultado = await self.result_repo.create_resultado(
            sesion_id=sesion.id,
            secuencia_texto=secuencia_texto,
            confianza_media=confianza_media,
            total_frames=len(frames),
        )

        detalles = []
        if detalles_data:
            detalles = await self.result_repo.create_detalles(
                resultado_id=resultado.id,
                detalles=detalles_data,
            )

        await self.db.commit()

        return {
            "sesion": SesionResponse.model_validate(sesion),
            "resultado": ResultadoResponse(
                id=resultado.id,
                sesion_id=resultado.sesion_id,
                secuencia_texto=resultado.secuencia_texto,
                confianza_media=resultado.confianza_media,
                total_frames=resultado.total_frames,
                detalles=[
                    DetalleResultadoResponse.model_validate(d)
                    for d in detalles
                ],
            ),
        }

    async def get_history(self, page: int = 1, limit: int = 20) -> dict:
        skip = (page - 1) * limit
        sesiones = await self.session_repo.get_by_usuario(
            usuario_id=self.usuario.id,
            skip=skip,
            limit=limit,
        )

        items = []
        for sesion in sesiones:
            resultado = sesion.resultado
            items.append({
                "sesion": SesionResponse.model_validate(sesion),
                "resultado": ResultadoResponse(
                    id=resultado.id,
                    sesion_id=resultado.sesion_id,
                    secuencia_texto=resultado.secuencia_texto,
                    confianza_media=resultado.confianza_media,
                    total_frames=resultado.total_frames,
                    detalles=[],  # historial no incluye detalles individuales
                ) if resultado else None,
            })

        return {"page": page, "limit": limit, "items": items}

    async def get_prediction_detail(self, sesion_id: int) -> dict:
        sesion = await self.session_repo.get_by_id(sesion_id)

        if not sesion or sesion.eliminado:
            raise HTTPException(status_code=404, detail="Sesión no encontrada.")

        if sesion.usuario_id != self.usuario.id:
            raise HTTPException(status_code=403, detail="Sin acceso a esta sesión.")

        resultado = sesion.resultado
        return {
            "sesion": SesionDetalleResponse.model_validate(sesion),
            "resultado": ResultadoResponse(
                id=resultado.id,
                sesion_id=resultado.sesion_id,
                secuencia_texto=resultado.secuencia_texto,
                confianza_media=resultado.confianza_media,
                total_frames=resultado.total_frames,
                detalles=[
                    DetalleResultadoResponse.model_validate(d)
                    for d in resultado.detalles
                ],
            ) if resultado else None,
        }
        
    async def delete_prediction(self, sesion_id: int) -> dict:
        sesion = await self.session_repo.get_by_id(sesion_id)

        if not sesion or sesion.eliminado:
            raise HTTPException(status_code=404, detail="Sesión no encontrada.")

        if sesion.usuario_id != self.usuario.id:
            raise HTTPException(status_code=403, detail="Sin acceso a esta sesión.")

        # Hard delete en Storage solo si la sesión tiene archivo asociado
        if sesion.archivo:
            await self.storage.delete_file(sesion.archivo.ruta_storage)

        # Soft delete en BD
        await self.session_repo.soft_delete(
            sesion=sesion,
            eliminado_por_id=self.usuario.id,
        )
        await self.db.commit()

        return {"message": "Sesión eliminada correctamente."}
    
    @staticmethod
    def _decodificar_imagen(contenido: bytes) -> np.ndarray:
        arr = np.frombuffer(contenido, dtype=np.uint8)
        imagen = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if imagen is None:
            raise HTTPException(
                status_code=422,
                detail="El archivo no es una imagen válida o está corrupto."
            )
        return imagen