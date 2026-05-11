# src/routers/predictions.py
from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.dependencies.auth import get_current_user
from src.models.session import ModoSesion
from src.models.user import Usuario
from src.repositories.session import SessionRepository
from src.services.predictions import PredictionsService
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