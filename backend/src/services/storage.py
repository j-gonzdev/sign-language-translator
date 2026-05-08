import uuid
from pathlib import Path

from fastapi import HTTPException
from supabase import create_client, Client

from src.core.config import get_settings

settings = get_settings()


_supabase_client = None

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    return _supabase_client


class StorageService:

    def __init__(self):
        self.client: Client = get_supabase_client()
        self.bucket: str = settings.SUPABASE_BUCKET


class StorageService:

    def __init__(self):
        self.client: Client = get_supabase_client()
        self.bucket: str = settings.SUPABASE_BUCKET

    def _build_path(self, usuario_id: int, sesion_id: int, nombre_original: str) -> str:
        ext = Path(nombre_original).suffix.lower()
        filename = f"{uuid.uuid4()}{ext}"
        return f"{usuario_id}/{sesion_id}/{filename}"

    async def upload_file(
        self,
        usuario_id: int,
        sesion_id: int,
        nombre_original: str,
        contenido: bytes,
        content_type: str,
    ) -> str:
        ruta = self._build_path(usuario_id, sesion_id, nombre_original)
        try:
            self.client.storage.from_(self.bucket).upload(
                path=ruta,
                file=contenido,
                file_options={"content-type": content_type},
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al subir archivo a Storage: {str(e)}"
            )
        return ruta

    async def get_signed_url(self, ruta_storage: str, expires_in: int = 3600) -> str:
        try:
            response = self.client.storage.from_(self.bucket).create_signed_url(
                path=ruta_storage,
                expires_in=expires_in,
            )
            return response["signedURL"]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al generar URL firmada: {str(e)}"
            )

    async def delete_file(self, ruta_storage: str) -> None:
        try:
            self.client.storage.from_(self.bucket).remove([ruta_storage])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al eliminar archivo de Storage: {str(e)}"
            )