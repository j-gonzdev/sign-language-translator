from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.dependencies.auth import get_current_user
from src.dependencies.roles import require_admin
from src.models.user import Usuario
from src.schemas.user import (
    PasswordUpdate,
    StatusUpdate,
    UsuarioResponse,
    UsuarioUpdate,
)
from src.services.users import UsersService

router = APIRouter()


@router.get("/me", response_model=UsuarioResponse)
async def get_me(current_user: Usuario = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UsuarioResponse)
async def update_me(
    data: UsuarioUpdate,
    current_user: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    return await service.update_me(current_user, data)


@router.put("/me/password", status_code=204)
async def update_password(
    data: PasswordUpdate,
    current_user: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    await service.update_password(current_user, data)


@router.delete("/me", status_code=204)
async def delete_own_account(
    current_user: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    await service.delete_own_account(current_user)


@router.get("", response_model=dict)
async def get_all_users(
    page: int = 1,
    limit: int = 20,
    current_user: Usuario = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    result = await service.get_all_users(page=page, limit=limit)
    return {
        "items": [UsuarioResponse.model_validate(u) for u in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "limit": result["limit"],
    }


@router.get("/{user_id}", response_model=UsuarioResponse)
async def get_user_by_id(
    user_id: int,
    current_user: Usuario = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    return await service.get_user_by_id(user_id)


@router.put("/{user_id}/status", response_model=UsuarioResponse)
async def update_user_status(
    user_id: int,
    data: StatusUpdate,
    current_user: Usuario = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    return await service.update_user_status(user_id, data, current_user)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    current_user: Usuario = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    service = UsersService(db)
    await service.delete_user(user_id, current_user)