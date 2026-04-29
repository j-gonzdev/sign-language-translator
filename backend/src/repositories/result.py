from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.result import DetalleResultado, Resultado


class ResultRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_resultado(self, sesion_id: int, secuencia_texto: str,
                                confianza_media: float,
                                total_frames: int) -> Resultado:
        resultado = Resultado(
            sesion_id=sesion_id,
            secuencia_texto=secuencia_texto,
            confianza_media=confianza_media,
            total_frames=total_frames,
        )
        self.db.add(resultado)
        await self.db.flush()
        await self.db.refresh(resultado)
        return resultado

    async def create_detalles(self, resultado_id: int,
                               detalles: list[dict]) -> list[DetalleResultado]:
        objetos = [
            DetalleResultado(
                resultado_id=resultado_id,
                gesto=d["gesto"],
                confianza=d["confianza"],
                posicion_secuencia=d["posicion_secuencia"],
                timestamp_frame=d.get("timestamp_frame"),
            )
            for d in detalles
        ]
        self.db.add_all(objetos)
        await self.db.flush()
        return objetos

    async def get_by_sesion(self, sesion_id: int) -> Resultado | None:
        result = await self.db.execute(
            select(Resultado).where(Resultado.sesion_id == sesion_id)
        )
        return result.scalar_one_or_none()