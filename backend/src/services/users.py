from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import hash_password, verify_password
from src.models.token import LogActividad
from src.models.user import UserStatus, Usuario
from src.repositories.user import UserRepository
from src.schemas.user import UsuarioUpdate, PasswordUpdate, StatusUpdate


class UsersService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserRepository(db)

    async def get_me(self, usuario: Usuario) -> Usuario:
        return usuario

    async def update_me(self, usuario: Usuario, data: UsuarioUpdate) -> Usuario:
        if data.nombre_usuario and data.nombre_usuario != usuario.nombre_usuario:
            existing = await self.repo.get_by_nombre_usuario(data.nombre_usuario)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="El nombre de usuario ya está en uso",
                )

        updates = data.model_dump(exclude_none=True)
        if not updates:
            return usuario

        return await self.repo.update(usuario, **updates)

    async def update_password(self, usuario: Usuario,
                               data: PasswordUpdate) -> None:
        if not verify_password(data.password_actual, usuario.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña actual es incorrecta",
            )

        if data.password_actual == data.password_nuevo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la actual",
            )

        await self.repo.update(
            usuario,
            password_hash=hash_password(data.password_nuevo)
        )

        await self._revoke_all_tokens(usuario.id)
        await self._log(usuario.id, "cambio_contrasena")
        await self.db.commit()

    async def delete_own_account(self, usuario: Usuario) -> None:
        await self._log(usuario.id, "eliminacion_cuenta")
        await self.repo.delete(usuario)
        await self.db.commit()

    async def get_all_users(self, page: int = 1, limit: int = 20) -> dict:
        skip = (page - 1) * limit
        usuarios = await self.repo.get_all(skip=skip, limit=limit)
        total = await self.repo.count_all()
        return {"items": usuarios, "total": total, "page": page, "limit": limit}

    async def get_user_by_id(self, user_id: int) -> Usuario:
        usuario = await self.repo.get_by_id(user_id)
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado",
            )
        return usuario

    async def update_user_status(self, user_id: int,
                                  data: StatusUpdate,
                                  admin: Usuario) -> Usuario:
        usuario = await self.get_user_by_id(user_id)

        if usuario.id == admin.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No puedes cambiar tu propio status",
            )
        
        if usuario.rol.nombre == "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No puedes cambiar el status de otro administrador",
            )

        await self.repo.update(usuario, status=data.status)

        if data.status == UserStatus.BANEADO:
            await self._revoke_all_tokens(user_id)

        await self._log(admin.id, "cambio_status")
        await self.db.commit()
        return usuario

    async def delete_user(self, user_id: int, admin: Usuario) -> None:
        usuario = await self.get_user_by_id(user_id)

        if usuario.id == admin.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No puedes eliminar tu propia cuenta desde el panel de admin",
            )
        
        if usuario.rol.nombre == "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No puedes eliminar a otro administrador",
            )

        await self._log(admin.id, "eliminacion_usuario")
        await self.repo.delete(usuario)
        await self.db.commit()

    async def _revoke_all_tokens(self, usuario_id: int) -> None:
        from sqlalchemy import update
        from src.models.token import RefreshToken
        await self.db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.usuario_id == usuario_id,
                RefreshToken.activo == True,
            )
            .values(activo=False)
        )

    async def _log(self, usuario_id: int, accion: str) -> None:
        log = LogActividad(usuario_id=usuario_id, accion=accion)
        self.db.add(log)
        await self.db.flush()