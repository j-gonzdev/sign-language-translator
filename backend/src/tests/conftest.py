import asyncpg
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

from src.main import app
from src.db import get_db

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/asl_test"
TEST_DATABASE_DSN = "postgresql://postgres:postgres@localhost:5432/asl_test"


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    conn = await asyncpg.connect(TEST_DATABASE_DSN)
    await conn.execute(
        "TRUNCATE TABLE detalle_resultado, resultado, archivo, "
        "sesion_traduccion, refresh_token, log_actividad, usuario "
        "RESTART IDENTITY CASCADE"
    )
    await conn.close()
    yield


@pytest_asyncio.fixture(scope="function")
async def client():
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def registered_user(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "user@test.com",
        "nombre_usuario": "testuser",
        "nombre": "Test",
        "apellidos": "User",
        "password": "12345678",
    })
    assert response.status_code == 201
    return response.json()


@pytest_asyncio.fixture(scope="function")
async def auth_headers(registered_user: dict):
    return {"Authorization": f"Bearer {registered_user['access_token']}"}