import cv2
import numpy as np
import asyncio
import base64
from jose import JWTError
from fastapi import HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from src.core.config import get_settings
from src.core.security import decode_token
from src.models.session import ModoSesion, SesionStatus
from src.models.user import Usuario, UserStatus
from src.repositories.session import SessionRepository
from src.repositories.result import ResultRepository
from src.repositories.user import UserRepository
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
    
    
WS_CLOSE_AUTH_ERROR   = 4001  # JWT ausente, expirado, inválido, o usuario no autorizado
WS_CLOSE_TIMEOUT      = 4002  # Límite de 180 s alcanzado — sesión COMPLETADA
WS_CLOSE_STOP_GESTURE = 4003  # Gesto de parada detectado — sesión COMPLETADA
WS_CLOSE_ML_ERROR     = 4004  # Excepción en modelo ML — sesión INTERRUMPIDA
    
    
class _StopGestureSignal(Exception):
    """Lanzada desde _live_loop cuando se detectan dos palmas abiertas."""
 
class _MLErrorSignal(Exception):
    """Lanzada desde _live_loop cuando el modelo ML lanza una excepción."""

async def predict_live(websocket: WebSocket, db) -> None:
    """
    Gestiona una sesión de traducción ASL en tiempo real via WebSocket.
 
    PROTOCOLO CLIENTE → SERVIDOR
    ─────────────────────────────────────────────────────────────────────
    Paso 1 — Autenticación (cliente envía nada más conectar):
        { "type": "auth", "token": "<access_jwt>" }
 
    Paso 2 — Confirmación del servidor:
        { "type": "auth_ok", "sesion_id": <int> }
        Si el token es inválido: cierre con código 4001.
 
    Paso 3 — El cliente envía frames a ~6 FPS:
        { "type": "frame", "data": "<base64_jpeg>" }
 
        Cómo capturar en Angular:
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const data = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
            ws.send(JSON.stringify({ type: 'frame', data }));
        Intervalo recomendado: setInterval(..., 150)  // ~6-7 FPS
 
    Paso 4 — El servidor responde por cada frame:
        { "type": "prediction", "gesto": "A", "confianza": 0.97, "secuencia": "HLA" }
        Si no hay mano: gesto = "nothing", confianza = 0.0
 
    CIERRES
    ─────────────────────────────────────────────────────────────────────
    4001  Token inválido / usuario baneado o inactivo   → sin sesión en BD
    4002  Timeout de 180 s                              → sesión COMPLETADA
    4003  Gesto de parada (dos palmas abiertas)         → sesión COMPLETADA
    4004  Excepción en modelo ML                        → sesión INTERRUMPIDA
    sin código  Desconexión abrupta del cliente         → sesión INTERRUMPIDA
 
    Args:
        websocket:  Instancia WebSocket de FastAPI (NO aceptada todavía).
        db:         AsyncSession de SQLAlchemy inyectada por FastAPI.
    """
    await websocket.accept()
 
    # ── 1. AUTENTICACIÓN ──────────────────────────────────────────────────────
    usuario = await _authenticate_websocket(websocket, db)
    if usuario is None:
        return  # ya cerró con 4001
 
    # ── 2. CREAR SESIÓN EN BD ─────────────────────────────────────────────────
    session_repo = SessionRepository(db)
    result_repo  = ResultRepository(db)
 
    sesion = await session_repo.create_sesion(
        usuario_id=usuario.id,
        modo=ModoSesion.LIVE_SESSION,
        status=SesionStatus.COMPLETADA,  # optimista; se sobrescribe si hay error
    )
    await db.commit()
 
    await websocket.send_json({
        "type": "auth_ok",
        "sesion_id": sesion.id,
    })
 
    # ── 3. ESTADO COMPARTIDO DEL BUCLE ────────────────────────────────────────
    # Usamos un dict mutable para que _live_loop pueda modificar el estado
    # sin necesitar nonlocal (asyncio.wait_for cancela la coroutine y nonlocal
    # no sobrevive a la cancelación de forma fiable entre implementaciones).
    state = {
        "ultimo_gesto": None,
        "detalles_data": [],
        "frames_procesados": 0,
    }
 
    # ── 4. BUCLE PRINCIPAL CON TIMEOUT ────────────────────────────────────────
    predictor = get_predictor()
    cierre_controlado = False
 
    try:
        await asyncio.wait_for(
            _live_loop(websocket=websocket, predictor=predictor, state=state),
            timeout=float(settings.VIDEO_MAX_DURATION),
        )
        # _live_loop solo termina "normalmente" si el iterador de mensajes
        # se agota (el cliente cerró limpiamente) — lo tratamos como INTERRUMPIDA
        # porque el usuario no usó el gesto de parada ni esperó el timeout.
        cierre_controlado = False
 
    except asyncio.TimeoutError:
        cierre_controlado = True
        await _safe_close(websocket, WS_CLOSE_TIMEOUT)
 
    except _StopGestureSignal:
        cierre_controlado = True
        await _safe_close(websocket, WS_CLOSE_STOP_GESTURE)
 
    except _MLErrorSignal:
        await session_repo.update_status(sesion, SesionStatus.INTERRUMPIDA)
        await db.commit()
        await _safe_close(websocket, WS_CLOSE_ML_ERROR)
        return
 
    except WebSocketDisconnect:
        cierre_controlado = False  # desconexión abrupta
 
    except Exception:
        await session_repo.update_status(sesion, SesionStatus.INTERRUMPIDA)
        await db.commit()
        return
 
    # ── 5. GUARDAR RESULTADO ──────────────────────────────────────────────────
    if not cierre_controlado:
        await session_repo.update_status(sesion, SesionStatus.INTERRUMPIDA)
        await db.commit()
        return
 
    detalles_data = state["detalles_data"]
    secuencia_texto = "".join(d["gesto"] for d in detalles_data)
    confianza_media = (
        sum(d["confianza"] for d in detalles_data) / len(detalles_data)
        if detalles_data else 0.0
    )
 
    resultado = await result_repo.create_resultado(
        sesion_id=sesion.id,
        secuencia_texto=secuencia_texto,
        confianza_media=confianza_media,
        total_frames=state["frames_procesados"],
    )
 
    if detalles_data:
        await result_repo.create_detalles(
            resultado_id=resultado.id,
            detalles=detalles_data,
        )
 
    await db.commit()

async def _live_loop(websocket: WebSocket, predictor, state: dict) -> None:
    """
    Recibe frames JSON del cliente en bucle, ejecuta ML y devuelve predicciones.
 
    El estado de deduplicación vive en `state` (dict mutable) para que
    predict_live() pueda leerlo después de que asyncio.wait_for lo cancele.
 
    Lanza:
        _StopGestureSignal   al detectar dos palmas abiertas
        _MLErrorSignal       si el modelo ML lanza una excepción no esperada
        WebSocketDisconnect  si el cliente se desconecta
    """
    async for message in websocket.iter_json():
        if message.get("type") != "frame":
            continue  # ignorar mensajes de control desconocidos
 
        frame = _decode_frame(message.get("data", ""))
        if frame is None:
            await websocket.send_json({
                "type": "error",
                "detail": "Frame inválido o corrupto, se ignora.",
            })
            continue
 
        state["frames_procesados"] += 1
 
        # ML — is_stop_gesture primero para no inferir gesto en el frame de parada
        try:
            if predictor.is_stop_gesture(frame):
                raise _StopGestureSignal()
            gesto, confianza = predictor.predict(frame)
        except (_StopGestureSignal, _MLErrorSignal):
            raise
        except Exception as e:
            raise _MLErrorSignal(str(e)) from e
 
        # Deduplicación — algoritmo idéntico a predict_video
        if gesto == "nothing":
            state["ultimo_gesto"] = None
        elif gesto != state["ultimo_gesto"]:
            state["ultimo_gesto"] = gesto
            state["detalles_data"].append({
                "gesto": gesto,
                "confianza": confianza,
                "posicion_secuencia": len(state["detalles_data"]),
                "timestamp_frame": None,  # live no tiene timestamp de vídeo
            })
 
        secuencia_actual = "".join(d["gesto"] for d in state["detalles_data"])
        await websocket.send_json({
            "type": "prediction",
            "gesto": gesto,
            "confianza": round(confianza, 4),
            "secuencia": secuencia_actual,
        })
        
async def _authenticate_websocket(websocket: WebSocket, db) -> "Usuario | None":
    """
    Valida el JWT enviado en el primer mensaje del WebSocket.
 
    No reutiliza get_current_user() de FastAPI porque esa dependency
    lee el header Authorization (HTTP Bearer), que no existe en WebSocket.
    Aquí el JWT viaja en el cuerpo del primer mensaje del protocolo.
 
    Devuelve el objeto Usuario si es válido, None si no (ya cerró el WS).
    Timeout de 10 s para el primer mensaje — evita conexiones fantasma.
    """
    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=10.0,
        )
    except (asyncio.TimeoutError, Exception):
        await _safe_close(websocket, WS_CLOSE_AUTH_ERROR)
        return None
 
    if message.get("type") != "auth":
        await _safe_close(websocket, WS_CLOSE_AUTH_ERROR)
        return None
 
    token = message.get("token", "")
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise JWTError("tipo de token incorrecto")
        user_id = payload.get("sub")
        if not user_id:
            raise JWTError("sub ausente")
    except JWTError:
        await _safe_close(websocket, WS_CLOSE_AUTH_ERROR)
        return None
 
    repo = UserRepository(db)
    usuario = await repo.get_by_id(int(user_id))
 
    if usuario is None:
        await _safe_close(websocket, WS_CLOSE_AUTH_ERROR)
        return None
 
    if usuario.status in (UserStatus.BANEADO, UserStatus.INACTIVO):
        await _safe_close(websocket, WS_CLOSE_AUTH_ERROR)
        return None
 
    return usuario
 
 
def _decode_frame(data: str) -> "np.ndarray | None":
    """
    Convierte un string base64 JPEG en un array numpy BGR (uint8).
 
    Devuelve None si el string está vacío, es base64 inválido,
    o los bytes resultantes no son decodificables por OpenCV.
    """
    if not data:
        return None
    try:
        raw   = base64.b64decode(data)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame  # puede ser None si OpenCV no reconoce el formato
    except Exception:
        return None
 
 
async def _safe_close(websocket: WebSocket, code: int) -> None:
    """
    Cierra el WebSocket ignorando errores si la conexión ya está cerrada.
    Necesario porque WebSocketDisconnect puede haber llegado justo antes.
    """
    try:
        await websocket.close(code=code)
    except Exception:
        pass