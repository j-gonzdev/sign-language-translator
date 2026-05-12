# src/routers/predictions.py
from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException, Query, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.dependencies.auth import get_current_user
from src.models.session import ModoSesion
from src.models.user import Usuario
from src.repositories.session import SessionRepository
from src.services.predictions import PredictionsService, predict_live
from src.services.storage import StorageService

router = APIRouter()


@router.post("/image")
async def predict_image(
    archivo: UploadFile = File(...),
    modo: ModoSesion = Form(...),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    service = PredictionsService(db=db, usuario=usuario)
    return await service.predict_image(archivo=archivo, modo=modo)


@router.post("/video")
async def predict_video(
    archivo: UploadFile = File(...),
    modo: ModoSesion = Form(...),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    service = PredictionsService(db=db, usuario=usuario)
    return await service.predict_video(archivo=archivo, modo=modo)

@router.websocket("/live")
async def predictions_live(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket para traducción ASL en tiempo real.
 
    La autenticación NO usa el header Authorization — el JWT viaja en el
    primer mensaje JSON tras establecer la conexión:
        { "type": "auth", "token": "<access_jwt>" }
 
    Ver predict_live() en services/predictions.py para el protocolo completo.
    """
    await predict_live(websocket=websocket, db=db)

@router.get("")
async def get_history(
    page: int = 1,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    service = PredictionsService(db=db, usuario=usuario)
    return await service.get_history(page=page, limit=limit)

@router.get("/files/{sesion_id}")
async def get_signed_url(
    sesion_id: int,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    session_repo = SessionRepository(db)
    sesion = await session_repo.get_by_id(sesion_id)

    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")

    if sesion.eliminado:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")

    if sesion.usuario_id != usuario.id:
        raise HTTPException(status_code=403, detail="Sin acceso a esta sesión.")

    if not sesion.archivo:
        raise HTTPException(status_code=404, detail="Esta sesión no tiene archivo asociado.")

    storage = StorageService()
    url = await storage.get_signed_url(sesion.archivo.ruta_storage)
    return {"url": url}

@router.get("/{sesion_id}")
async def get_prediction_detail(
    sesion_id: int,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    service = PredictionsService(db=db, usuario=usuario)
    return await service.get_prediction_detail(sesion_id=sesion_id)

@router.delete("/{sesion_id}")
async def delete_prediction(
    sesion_id: int,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
):
    service = PredictionsService(db=db, usuario=usuario)
    return await service.delete_prediction(sesion_id=sesion_id)