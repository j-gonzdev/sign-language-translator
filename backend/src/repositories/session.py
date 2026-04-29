from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.session import Archivo, ModoSesion, SesionTraduccion
from src.models.result import Resultado


class SessionRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_sesion(self, usuario_id: int, modo: ModoSesion) -> SesionTraduccion:
        sesion = SesionTraduccion(
            usuario_id=usuario_id,
            modo=modo,
            eliminado=False,
        )
        self.db.add(sesion)
        await self.db.flush()
        await self.db.refresh(sesion)
        return sesion

    async def get_by_id(self, sesion_id: int) -> SesionTraduccion | None:
        result = await self.db.execute(
            select(SesionTraduccion).where(SesionTraduccion.id == sesion_id)
        )
        return result.scalar_one_or_none()

    async def get_by_usuario(self, usuario_id: int, skip: int = 0,
                              limit: int = 20) -> list[SesionTraduccion]:
        result = await self.db.execute(
            select(SesionTraduccion)
            .where(
                SesionTraduccion.usuario_id == usuario_id,
                SesionTraduccion.eliminado == False,
            )
            .order_by(SesionTraduccion.fecha.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_all(self, skip: int = 0, limit: int = 20) -> list[SesionTraduccion]:
        result = await self.db.execute(
            select(SesionTraduccion)
            .order_by(SesionTraduccion.fecha.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def soft_delete(self, sesion: SesionTraduccion,
                          eliminado_por_id: int) -> SesionTraduccion:
        sesion.eliminado = True
        sesion.fecha_eliminacion = datetime.utcnow()
        sesion.eliminado_por_id = eliminado_por_id
        await self.db.flush()
        return sesion

    async def create_archivo(self, sesion_id: int, nombre_original: str,
                              ruta_storage: str, formato: str,
                              tamano: int) -> Archivo:
        archivo = Archivo(
            sesion_id=sesion_id,
            nombre_original=nombre_original,
            ruta_storage=ruta_storage,
            formato=formato,
            tamano=tamano,
        )
        self.db.add(archivo)
        await self.db.flush()
        await self.db.refresh(archivo)
        return archivo