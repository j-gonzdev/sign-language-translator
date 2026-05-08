from functools import lru_cache
from pydantic import field_validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Entorno
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # Base de datos
    DATABASE_URL: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Supabase Storage
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_BUCKET: str = "asl-files"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:4200"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Administración
    ADMIN_SECRET: str
    
    MODEL_PATH: str = "ml/models/asl_model.pkl"
    TASK_PATH: str = "ml/hand_landmarker.task"

    # Propiedades calculadas
    @property
    def allowed_origins_list(self) -> list[str]:
        """Convierte el string CSV en lista para FastAPI CORS."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    # Validadores
    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def jwt_secret_must_be_strong(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY debe tener al menos 32 caracteres. "
                "Genera uno con: openssl rand -hex 32"
            )
        return v

    @field_validator("APP_ENV")
    @classmethod
    def app_env_must_be_valid(cls, v: str) -> str:
        allowed = {"development", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV debe ser uno de: {allowed}")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()