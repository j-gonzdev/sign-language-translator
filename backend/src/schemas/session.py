from datetime import datetime
from pydantic import BaseModel
from src.models.session import ModoSesion


class SesionResponse(BaseModel):
    id: int
    usuario_id: int
    modo: ModoSesion
    fecha: datetime
    eliminado: bool

    model_config = {"from_attributes": True}


class SesionDetalleResponse(SesionResponse):
    fecha_eliminacion: datetime | None
    eliminado_por_id: int | None