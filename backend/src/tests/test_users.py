import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_obtener_perfil_correcto(client: AsyncClient, auth_headers: dict):
    response = await client.get("/users/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@test.com"
    assert data["nombre_usuario"] == "testuser"
    assert "password_hash" not in data


async def test_actualizar_perfil_correcto(client: AsyncClient, auth_headers: dict):
    response = await client.put("/users/me", headers=auth_headers, json={
        "nombre": "Nuevo Nombre",
    })
    assert response.status_code == 200
    assert response.json()["nombre"] == "Nuevo Nombre"


async def test_actualizar_nombre_usuario_duplicado(client: AsyncClient, auth_headers: dict):
    await client.post("/auth/register", json={
        "email": "otro@test.com",
        "nombre_usuario": "otrousuario",
        "nombre": "Otro",
        "apellidos": "Usuario",
        "password": "12345678",
    })
    response = await client.put("/users/me", headers=auth_headers, json={
        "nombre_usuario": "otrousuario",
    })
    assert response.status_code == 409


async def test_cambiar_contrasena_correcto(client: AsyncClient, auth_headers: dict):
    response = await client.put("/users/me/password", headers=auth_headers, json={
        "password_actual": "12345678",
        "password_nuevo": "nuevapassword123",
    })
    assert response.status_code == 204


async def test_cambiar_contrasena_incorrecta(client: AsyncClient, auth_headers: dict):
    response = await client.put("/users/me/password", headers=auth_headers, json={
        "password_actual": "wrongpassword",
        "password_nuevo": "nuevapassword123",
    })
    assert response.status_code == 400


async def test_eliminar_cuenta_propia(client: AsyncClient, auth_headers: dict):
    response = await client.delete("/users/me", headers=auth_headers)
    assert response.status_code == 204

    me_response = await client.get("/users/me", headers=auth_headers)
    assert me_response.status_code == 401


async def test_obtener_lista_usuarios_siendo_admin(
    client: AsyncClient,
    registered_user: dict,
    admin_headers: dict,
):
    response = await client.get("/users", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


async def test_obtener_lista_usuarios_siendo_user(client: AsyncClient, auth_headers: dict):
    response = await client.get("/users", headers=auth_headers)
    assert response.status_code == 403


async def test_obtener_usuario_por_id_siendo_user(client: AsyncClient,
                                                   registered_user: dict,
                                                   auth_headers: dict):
    response = await client.get("/users/1", headers=auth_headers)
    assert response.status_code == 403


async def test_eliminar_usuario_siendo_user(client: AsyncClient, auth_headers: dict):
    response = await client.delete("/users/1", headers=auth_headers)
    assert response.status_code == 403


async def test_obtener_usuario_por_id_siendo_admin(
    client: AsyncClient,
    registered_user: dict,
    admin_headers: dict,
):
    user_id = registered_user["id"]
    response = await client.get(f"/users/{user_id}", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["email"] == "user@test.com"


async def test_actualizar_status_usuario_correcto(
    client: AsyncClient,
    registered_user: dict,
    admin_headers: dict,
):
    user_id = registered_user["id"]
    user_token = registered_user["access_token"]

    response = await client.put(
        f"/users/{user_id}/status",
        json={"status": "BANEADO"},
        headers=admin_headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "BANEADO"

    response = await client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert response.status_code == 403


async def test_eliminar_usuario_siendo_admin(
    client: AsyncClient,
    registered_user: dict,
    admin_headers: dict,
):
    user_id = registered_user["id"]

    response = await client.delete(
        f"/users/{user_id}", headers=admin_headers
    )
    assert response.status_code == 204

    response = await client.get(
        f"/users/{user_id}", headers=admin_headers
    )
    assert response.status_code == 404