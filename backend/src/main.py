from contextlib import asynccontextmanager

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


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ejecuta setup al arrancar y cleanup al apagar."""
    setup_logging()
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