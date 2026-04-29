from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.dependencies.auth import get_current_user
from src.models.user import Usuario
from src.schemas.token import LoginRequest, LogoutRequest, RefreshRequest, TokenResponse
from src.schemas.user import UsuarioCreate, UsuarioResponse
from src.services.auth import AuthService

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    data: UsuarioCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    service = AuthService(db)
    return await service.register(
        email=data.email,
        password=data.password,
        nombre_usuario=data.nombre_usuario,
        nombre=data.nombre,
        apellidos=data.apellidos,
        request=request,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    service = AuthService(db)
    return await service.login(
        email=data.email,
        password=data.password,
        request=request,
    )


@router.post("/logout", status_code=204)
async def logout(
    data: LogoutRequest,
    request: Request,
    current_user: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = AuthService(db)
    await service.logout(
        refresh_token=data.refresh_token,
        usuario_id=current_user.id,
        request=request,
    )


@router.post("/logout-all", status_code=204)
async def logout_all(
    request: Request,
    current_user: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = AuthService(db)
    await service.revoke_all_tokens(
        usuario_id=current_user.id,
        request=request,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    data: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    service = AuthService(db)
    return await service.refresh(refresh_token=data.refresh_token)