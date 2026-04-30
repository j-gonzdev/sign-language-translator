import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db import Base


class UserStatus(str, enum.Enum):
    ACTIVO = "activo"
    INACTIVO = "inactivo"
    BANEADO = "baneado"


class Rol(Base):
    __tablename__ = "rol"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)

    usuarios: Mapped[list["Usuario"]] = relationship(
        "Usuario", back_populates="rol", lazy="selectin"
    )


class Usuario(Base):
    __tablename__ = "usuario"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rol_id: Mapped[int] = mapped_column(Integer, ForeignKey("rol.id"), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nombre_usuario: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    apellidos: Mapped[str] = mapped_column(String(200), nullable=False)
    fecha_registro: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    fecha_ultimo_acceso: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, name="userstatus"), nullable=False, default=UserStatus.ACTIVO
    )

    rol: Mapped[Rol] = relationship("Rol", back_populates="usuarios", lazy="selectin")