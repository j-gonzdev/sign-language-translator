import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db import Base


class ModoSesion(str, enum.Enum):
    IMAGEN_SUBIDA = "imagen_subida"
    FOTO_CAPTURADA = "foto_capturada"
    VIDEO_SUBIDO = "video_subido"
    VIDEO_GRABADO = "video_grabado"
    LIVE_SESSION = "live_session"


class SesionTraduccion(Base):
    __tablename__ = "sesion_traduccion"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    usuario_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("usuario.id", ondelete="CASCADE"), nullable=False, index=True
    )
    modo: Mapped[ModoSesion] = mapped_column(
        Enum(ModoSesion, name="modosesion"), nullable=False
    )
    fecha: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    eliminado: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fecha_eliminacion: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    eliminado_por_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("usuario.id"), nullable=True
    )

    usuario: Mapped["Usuario"] = relationship(
        "Usuario", back_populates="sesiones", lazy="selectin",
        foreign_keys=[usuario_id]
    )
    archivo: Mapped["Archivo"] = relationship(
        "Archivo", back_populates="sesion", lazy="selectin", uselist=False
    )
    resultado: Mapped["Resultado"] = relationship(
        "Resultado", back_populates="sesion", lazy="selectin", uselist=False
    )


class Archivo(Base):
    __tablename__ = "archivo"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sesion_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sesion_traduccion.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    nombre_original: Mapped[str] = mapped_column(String(255), nullable=False)
    ruta_storage: Mapped[str] = mapped_column(String(500), nullable=False)
    formato: Mapped[str] = mapped_column(String(50), nullable=False)
    tamano: Mapped[int] = mapped_column(Integer, nullable=False)
    fecha_subida: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    sesion: Mapped[SesionTraduccion] = relationship(
        "SesionTraduccion", back_populates="archivo", lazy="selectin"
    )