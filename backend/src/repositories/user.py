from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import Rol, Usuario, UserStatus


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Usuario | None:
        result = await self.db.execute(
            select(Usuario).where(Usuario.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Usuario | None:
        result = await self.db.execute(
            select(Usuario).where(Usuario.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_nombre_usuario(self, nombre_usuario: str) -> Usuario | None:
        result = await self.db.execute(
            select(Usuario).where(Usuario.nombre_usuario == nombre_usuario)
        )
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 20) -> list[Usuario]:
        result = await self.db.execute(
            select(Usuario).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_all(self) -> int:
        from sqlalchemy import func
        result = await self.db.execute(
            select(func.count()).select_from(Usuario)
        )
        return result.scalar_one()

    async def create(self, email: str, password_hash: str, nombre_usuario: str,
                     nombre: str, apellidos: str, rol_id: int) -> Usuario:
        usuario = Usuario(
            email=email,
            password_hash=password_hash,
            nombre_usuario=nombre_usuario,
            nombre=nombre,
            apellidos=apellidos,
            rol_id=rol_id,
            status=UserStatus.ACTIVO,
        )
        self.db.add(usuario)
        await self.db.flush()
        await self.db.refresh(usuario)
        return usuario

    async def update(self, usuario: Usuario, **kwargs) -> Usuario:
        for key, value in kwargs.items():
            setattr(usuario, key, value)
        await self.db.flush()
        await self.db.refresh(usuario)
        return usuario

    async def delete(self, usuario: Usuario) -> None:
        await self.db.delete(usuario)
        await self.db.flush()

    async def get_rol_by_nombre(self, nombre: str) -> Rol | None:
        result = await self.db.execute(
            select(Rol).where(Rol.nombre == nombre)
        )
        return result.scalar_one_or_none()