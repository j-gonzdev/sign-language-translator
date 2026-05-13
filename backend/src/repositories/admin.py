# backend/src/repositories/admin.py
 
from datetime import datetime, timedelta, timezone
 
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
 
from src.models.session import ModoSesion, SesionTraduccion
from src.models.result import DetalleResultado
from src.models.user import Usuario
from src.models.token import LogActividad
 
 
class AdminRepository:
 
    def __init__(self, db: AsyncSession):
        self.db = db
 
    # ── Gestión ───────────────────────────────────────────────────────────────
 
    async def get_all_users(self, skip: int = 0, limit: int = 20) -> list[Usuario]:
        result = await self.db.execute(
            select(Usuario)
            .order_by(Usuario.fecha_registro.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
 
    async def count_users(self) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(Usuario)
        )
        return result.scalar_one()
 
    async def get_all_predictions(
        self, skip: int = 0, limit: int = 20
    ) -> list[SesionTraduccion]:
        result = await self.db.execute(
            select(SesionTraduccion)
            .order_by(SesionTraduccion.fecha.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
 
    async def count_predictions(self) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(SesionTraduccion)
        )
        return result.scalar_one()
 
    async def get_logs(self, skip: int = 0, limit: int = 50) -> list[LogActividad]:
        result = await self.db.execute(
            select(LogActividad)
            .order_by(LogActividad.fecha.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
 
    async def count_logs(self) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(LogActividad)
        )
        return result.scalar_one()
 
    # ── Estadísticas ──────────────────────────────────────────────────────────
 
    async def get_stats_gestures(self, limit: int = 10) -> list[dict]:
        """
        Gestos más detectados.
        COUNT agrupado por gesto sobre DetalleResultado,
        ordenado de mayor a menor frecuencia.
        """
        result = await self.db.execute(
            select(
                DetalleResultado.gesto,
                func.count(DetalleResultado.id).label("total"),
                func.avg(DetalleResultado.confianza).label("confianza_media"),
            )
            .group_by(DetalleResultado.gesto)
            .order_by(func.count(DetalleResultado.id).desc())
            .limit(limit)
        )
        return [
            {
                "gesto": row.gesto,
                "total": row.total,
                "confianza_media": round(row.confianza_media, 4),
            }
            for row in result.all()
        ]
 
    async def get_stats_users(self, limit: int = 10) -> list[dict]:
        """
        Usuarios más activos.
        COUNT de sesiones no eliminadas por usuario,
        ordenado de mayor a menor.
        """
        result = await self.db.execute(
            select(
                SesionTraduccion.usuario_id,
                Usuario.nombre_usuario,
                Usuario.email,
                func.count(SesionTraduccion.id).label("total_sesiones"),
            )
            .join(Usuario, SesionTraduccion.usuario_id == Usuario.id)
            .where(SesionTraduccion.eliminado == False)
            .group_by(
                SesionTraduccion.usuario_id,
                Usuario.nombre_usuario,
                Usuario.email,
            )
            .order_by(func.count(SesionTraduccion.id).desc())
            .limit(limit)
        )
        return [
            {
                "usuario_id": row.usuario_id,
                "nombre_usuario": row.nombre_usuario,
                "email": row.email,
                "total_sesiones": row.total_sesiones,
            }
            for row in result.all()
        ]
 
    async def get_stats_activity(self, periodo: str) -> list[dict]:
        """
        Actividad por periodo.
        periodo='day'   → últimas 24h agrupadas por hora
        periodo='week'  → últimos 7 días agrupados por día
        periodo='month' → últimos 30 días agrupados por día
        """
        now = datetime.now(timezone.utc)
 
        if periodo == "day":
            desde = now - timedelta(hours=24)
            trunc = func.date_trunc("hour", SesionTraduccion.fecha)
        elif periodo == "week":
            desde = now - timedelta(days=7)
            trunc = func.date_trunc("day", SesionTraduccion.fecha)
        else:  # month
            desde = now - timedelta(days=30)
            trunc = func.date_trunc("day", SesionTraduccion.fecha)
 
        result = await self.db.execute(
            select(
                trunc.label("intervalo"),
                func.count(SesionTraduccion.id).label("total"),
            )
            .where(SesionTraduccion.fecha >= desde)
            .group_by(trunc)
            .order_by(trunc)
        )
        return [
            {
                "intervalo": row.intervalo.isoformat(),
                "total": row.total,
            }
            for row in result.all()
        ]
 
    async def get_stats_modes(self) -> list[dict]:
        """
        Distribución por modo.
        Incluye todos los modos aunque tengan 0 sesiones.
        """
        result = await self.db.execute(
            select(
                SesionTraduccion.modo,
                func.count(SesionTraduccion.id).label("total"),
            )
            .group_by(SesionTraduccion.modo)
        )
        rows = {row.modo: row.total for row in result.all()}
 
        return [
            {
                "modo": modo.value,
                "total": rows.get(modo, 0),
            }
            for modo in ModoSesion
        ]
 
    async def get_stats_confidence(self) -> list[dict]:
        """
        Confianza media por gesto.
        AVG agrupado sobre DetalleResultado, ordenado de mayor a menor.
        """
        result = await self.db.execute(
            select(
                DetalleResultado.gesto,
                func.avg(DetalleResultado.confianza).label("confianza_media"),
                func.count(DetalleResultado.id).label("total_muestras"),
            )
            .group_by(DetalleResultado.gesto)
            .order_by(func.avg(DetalleResultado.confianza).desc())
        )
        return [
            {
                "gesto": row.gesto,
                "confianza_media": round(row.confianza_media, 4),
                "total_muestras": row.total_muestras,
            }
            for row in result.all()
        ]
 
    async def get_stats_registrations(self, days: int = 30) -> list[dict]:
        """
        Registros de usuarios en el tiempo.
        COUNT agrupado por día, últimos `days` días, orden cronológico.
        """
        desde = datetime.now(timezone.utc) - timedelta(days=days)
        trunc = func.date_trunc("day", Usuario.fecha_registro)
 
        result = await self.db.execute(
            select(
                trunc.label("dia"),
                func.count(Usuario.id).label("total"),
            )
            .where(Usuario.fecha_registro >= desde)
            .group_by(trunc)
            .order_by(trunc)
        )
        return [
            {
                "dia": row.dia.isoformat(),
                "total": row.total,
            }
            for row in result.all()
        ]