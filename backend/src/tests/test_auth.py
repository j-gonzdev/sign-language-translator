import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_register_correcto(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "nuevo@test.com",
        "nombre_usuario": "nuevousuario",
        "nombre": "Nuevo",
        "apellidos": "Usuario",
        "password": "12345678",
    })
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


async def test_register_email_duplicado(client: AsyncClient, registered_user: dict):
    response = await client.post("/auth/register", json={
        "email": "user@test.com",
        "nombre_usuario": "otrousuario",
        "nombre": "Otro",
        "apellidos": "Usuario",
        "password": "12345678",
    })
    assert response.status_code == 409
    assert "email" in response.json()["detail"].lower()


async def test_register_nombre_usuario_duplicado(client: AsyncClient, registered_user: dict):
    response = await client.post("/auth/register", json={
        "email": "otro@test.com",
        "nombre_usuario": "testuser",
        "nombre": "Otro",
        "apellidos": "Usuario",
        "password": "12345678",
    })
    assert response.status_code == 409
    assert "usuario" in response.json()["detail"].lower()


async def test_login_correcto(client: AsyncClient, registered_user: dict):
    response = await client.post("/auth/login", json={
        "email": "user@test.com",
        "password": "12345678",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


async def test_login_credenciales_incorrectas(client: AsyncClient, registered_user: dict):
    response = await client.post("/auth/login", json={
        "email": "user@test.com",
        "password": "wrongpassword",
    })
    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciales incorrectas"


async def test_login_email_inexistente(client: AsyncClient):
    response = await client.post("/auth/login", json={
        "email": "noexiste@test.com",
        "password": "12345678",
    })
    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciales incorrectas"


async def test_logout_correcto(client: AsyncClient, registered_user: dict, auth_headers: dict):
    response = await client.post("/auth/logout",
        json={"refresh_token": registered_user["refresh_token"]},
        headers=auth_headers,
    )
    assert response.status_code == 204


async def test_logout_token_invalido(client: AsyncClient, auth_headers: dict):
    response = await client.post("/auth/logout",
        json={"refresh_token": "token.invalido.aqui"},
        headers=auth_headers,
    )
    assert response.status_code == 204


async def test_refresh_correcto(client: AsyncClient, registered_user: dict):
    response = await client.post("/auth/refresh", json={
        "refresh_token": registered_user["refresh_token"],
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] != registered_user["refresh_token"]


async def test_refresh_token_expirado_o_invalido(client: AsyncClient):
    response = await client.post("/auth/refresh", json={
        "refresh_token": "token.invalido.aqui",
    })
    assert response.status_code == 401


async def test_refresh_token_rotation(client: AsyncClient, registered_user: dict):
    first_refresh = await client.post("/auth/refresh", json={
        "refresh_token": registered_user["refresh_token"],
    })
    assert first_refresh.status_code == 200
    old_token = registered_user["refresh_token"]

    second_refresh = await client.post("/auth/refresh", json={
        "refresh_token": old_token,
    })
    assert second_refresh.status_code == 401


async def test_logout_all_devices(client: AsyncClient, registered_user: dict, auth_headers: dict):
    response = await client.post("/auth/logout-all", headers=auth_headers)
    assert response.status_code == 204

    refresh_response = await client.post("/auth/refresh", json={
        "refresh_token": registered_user["refresh_token"],
    })
    assert refresh_response.status_code == 401