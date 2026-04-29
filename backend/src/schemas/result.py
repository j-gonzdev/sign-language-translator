from pydantic import BaseModel
from src.models.session import ModoSesion


class DetalleResultadoResponse(BaseModel):
    id: int
    gesto: str
    confianza: float
    posicion_secuencia: int
    timestamp_frame: float | None

    model_config = {"from_attributes": True}


class ResultadoResponse(BaseModel):
    id: int
    sesion_id: int
    secuencia_texto: str
    confianza_media: float
    total_frames: int
    detalles: list[DetalleResultadoResponse]

    model_config = {"from_attributes": True}


class PrediccionDetalleResponse(BaseModel):
    sesion: SesionDetalleResponse
    resultado: ResultadoResponse | None
    modo: ModoSesion

    model_config = {"from_attributes": True}


from src.schemas.session import SesionDetalleResponse