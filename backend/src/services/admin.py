# backend/src/services/admin.py

from src.models.user import Usuario
from src.repositories.admin import AdminRepository
from src.schemas.user import UsuarioResponse


class AdminService:

    def __init__(self, db, usuario: Usuario):
        self.db = db
        self.usuario = usuario
        self.repo = AdminRepository(db)

    # ── Gestión ───────────────────────────────────────────────────────────────

    async def get_all_users(self, page: int = 1, limit: int = 20) -> dict:
        skip = (page - 1) * limit
        usuarios = await self.repo.get_all_users(skip=skip, limit=limit)
        total = await self.repo.count_users()
        return {
            "page": page,
            "limit": limit,
            "total": total,
            "items": [
                UsuarioResponse.model_validate(u).model_dump()
                for u in usuarios
            ],
        }

    async def get_all_predictions(self, page: int = 1, limit: int = 20) -> dict:
        from src.schemas.session import SesionResponse
        from src.schemas.result import ResultadoResponse

        skip = (page - 1) * limit
        sesiones = await self.repo.get_all_predictions(skip=skip, limit=limit)
        total = await self.repo.count_predictions()

        items = []
        for sesion in sesiones:
            resultado = sesion.resultado
            items.append({
                "sesion": SesionResponse.model_validate(sesion).model_dump(),
                "resultado": ResultadoResponse(
                    id=resultado.id,
                    sesion_id=resultado.sesion_id,
                    secuencia_texto=resultado.secuencia_texto,
                    confianza_media=resultado.confianza_media,
                    total_frames=resultado.total_frames,
                    detalles=[],
                ).model_dump() if resultado else None,
            })

        return {
            "page": page,
            "limit": limit,
            "total": total,
            "items": items,
        }

    async def get_logs(self, page: int = 1, limit: int = 50) -> dict:
        skip = (page - 1) * limit
        logs = await self.repo.get_logs(skip=skip, limit=limit)
        total = await self.repo.count_logs()
        return {
            "page": page,
            "limit": limit,
            "total": total,
            "items": [
                {
                    "id": log.id,
                    "usuario_id": log.usuario_id,
                    "accion": log.accion,
                    "fecha": log.fecha.isoformat(),
                    "ip": log.ip,
                }
                for log in logs
            ],
        }

    # ── Estadísticas ──────────────────────────────────────────────────────────

    async def get_stats_gestures(self) -> list[dict]:
        return await self.repo.get_stats_gestures()

    async def get_stats_users(self) -> list[dict]:
        return await self.repo.get_stats_users()

    async def get_stats_activity(self, periodo: str) -> list[dict]:
        periodos_validos = {"day", "week", "month"}
        if periodo not in periodos_validos:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"Periodo inválido. Usa: {sorted(periodos_validos)}",
            )
        return await self.repo.get_stats_activity(periodo=periodo)

    async def get_stats_modes(self) -> list[dict]:
        return await self.repo.get_stats_modes()

    async def get_stats_confidence(self) -> list[dict]:
        return await self.repo.get_stats_confidence()

    async def get_stats_registrations(self) -> list[dict]:
        return await self.repo.get_stats_registrations()