# backend/src/routers/admin.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.dependencies.auth import get_current_user
from src.dependencies.roles import require_admin
from src.models.user import Usuario
from src.services.admin import AdminService

router = APIRouter()


def _service(db: AsyncSession, usuario: Usuario) -> AdminService:
    return AdminService(db=db, usuario=usuario)


# ── Gestión ───────────────────────────────────────────────────────────────────

@router.get("/users")
async def get_all_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_all_users(page=page, limit=limit)


@router.get("/predictions")
async def get_all_predictions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_all_predictions(page=page, limit=limit)


@router.get("/logs")
async def get_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_logs(page=page, limit=limit)


# ── Estadísticas ──────────────────────────────────────────────────────────────

@router.get("/stats/gestures")
async def stats_gestures(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_gestures()


@router.get("/stats/users")
async def stats_users(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_users()


@router.get("/stats/activity")
async def stats_activity(
    periodo: str = Query("week", pattern="^(day|week|month)$"),
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_activity(periodo=periodo)


@router.get("/stats/modes")
async def stats_modes(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_modes()


@router.get("/stats/confidence")
async def stats_confidence(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_confidence()


@router.get("/stats/registrations")
async def stats_registrations(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(require_admin),
):
    return await _service(db, usuario).get_stats_registrations()