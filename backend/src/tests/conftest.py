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
    tokens = response.json()

    me_response = await client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert me_response.status_code == 200

    return {**tokens, **me_response.json()}


@pytest_asyncio.fixture(scope="function")
async def auth_headers(registered_user: dict):
    return {"Authorization": f"Bearer {registered_user['access_token']}"}

@pytest_asyncio.fixture(scope="function")
async def admin_user(client: AsyncClient):
    conn = await asyncpg.connect(TEST_DATABASE_DSN)

    admin_rol_id = await conn.fetchval(
        "SELECT id FROM rol WHERE nombre = 'admin'"
    )

    from src.core.security import hash_password
    password_hash = hash_password("adminpass123")

    await conn.execute(
        """
        INSERT INTO usuario (rol_id, email, password_hash, nombre_usuario, nombre, apellidos, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7::userstatus)
        """,
        admin_rol_id,
        "admin@test.com",
        password_hash,
        "adminuser",
        "Admin",
        "Test",
        "ACTIVO",
    )
    await conn.close()

    response = await client.post("/auth/login", json={
        "email": "admin@test.com",
        "password": "adminpass123",
    })
    assert response.status_code == 200
    return response.json()


@pytest_asyncio.fixture(scope="function")
async def admin_headers(admin_user: dict):
    return {"Authorization": f"Bearer {admin_user['access_token']}"}