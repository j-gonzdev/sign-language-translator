from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db import Base


class Resultado(Base):
    __tablename__ = "resultado"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sesion_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sesion_traduccion.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    secuencia_texto: Mapped[str] = mapped_column(Text, nullable=False)
    confianza_media: Mapped[float] = mapped_column(Float, nullable=False)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False)

    sesion: Mapped["SesionTraduccion"] = relationship(
        "SesionTraduccion", back_populates="resultado", lazy="selectin"
    )
    detalles: Mapped[list["DetalleResultado"]] = relationship(
        "DetalleResultado", back_populates="resultado", lazy="selectin"
    )


class DetalleResultado(Base):
    __tablename__ = "detalle_resultado"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    resultado_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("resultado.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    gesto: Mapped[str] = mapped_column(String(50), nullable=False)
    confianza: Mapped[float] = mapped_column(Float, nullable=False)
    posicion_secuencia: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_frame: Mapped[float | None] = mapped_column(Float, nullable=True)

    resultado: Mapped[Resultado] = relationship(
        "Resultado", back_populates="detalles", lazy="selectin"
    )