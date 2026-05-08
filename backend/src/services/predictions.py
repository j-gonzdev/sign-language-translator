import io
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
from src.schemas.session import SesionResponse

settings = get_settings()

# Modos válidos para el endpoint de imagen
_MODOS_IMAGEN = {ModoSesion.IMAGEN_SUBIDA, ModoSesion.FOTO_CAPTURADA}

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
        ruta_storage = await self.storage.upload_file(
            usuario_id=self.usuario.id,
            sesion_id=sesion.id,
            nombre_original=archivo.filename or "imagen.jpg",
            contenido=contenido,
            content_type=archivo.content_type or "image/jpeg",
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