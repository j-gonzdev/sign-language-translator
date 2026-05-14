# backend/src/main.py

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.routers import auth, users, predictions, admin

settings = get_settings()


def _download_model_if_needed():
    """
    Descarga el modelo .pkl desde Hugging Face Hub si no existe en disco.

    En desarrollo el modelo ya está en ml/models/asl_model.pkl (Git LFS local).
    En producción (Railway) el archivo no existe en la imagen — se descarga
    en el primer arranque y queda en disco para los siguientes reinicios
    del mismo contenedor.

    La descarga solo ocurre si el archivo no existe O si pesa menos de 1MB
    (lo que indicaría que es el puntero LFS en lugar del modelo real).
    """
    model_path = settings.MODEL_PATH
    MIN_VALID_SIZE = 1 * 1024 * 1024  # 1 MB — el puntero LFS pesa ~130 bytes

    model_exists = os.path.exists(model_path)
    model_valid = model_exists and os.path.getsize(model_path) > MIN_VALID_SIZE

    if model_valid:
        return  # modelo real ya en disco, no hay que descargar nada

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "El modelo no está en disco y HF_TOKEN no está configurado. "
            "Añade HF_TOKEN como variable de entorno en Railway."
        )

    from huggingface_hub import hf_hub_download
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Descargando modelo desde Hugging Face Hub...")

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id="j-gonzdev/asl-model",
        filename="asl_model.pkl",
        token=hf_token,
        local_dir=os.path.dirname(model_path),
    )

    logger.info(f"Modelo descargado en {downloaded_path} ({os.path.getsize(downloaded_path) / 1024 / 1024:.1f} MB)")


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ejecuta setup al arrancar y cleanup al apagar."""
    setup_logging()

    # Descargar modelo si no está en disco (producción) o ya existe (desarrollo)
    _download_model_if_needed()

    from src.services.predictions import get_predictor
    from src.services.storage import get_supabase_client
    client = get_supabase_client()
    client.storage.from_(settings.SUPABASE_BUCKET).list()  # calentamiento
    get_predictor()
    yield


# Instancia FastAPI
app = FastAPI(
    title="Sign Language Translator API",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)


# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(auth.router,        prefix="/auth",        tags=["Auth"])
app.include_router(users.router,       prefix="/users",       tags=["Users"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(admin.router,       prefix="/admin",       tags=["Admin"])


# Health
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "env": settings.APP_ENV}