from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings

settings = get_settings()

# Motor
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=not settings.is_production,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Fabrica de sesiones
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Base para modelos
class Base(DeclarativeBase):
    pass


# Dependencia FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise