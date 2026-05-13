# backend/src/tests/test_admin.py
"""
Tests para los endpoints de administración.
 
Estrategia:
- BD real (asl_test) — se insertan datos desde los fixtures existentes
- Sin mocks — los endpoints de admin leen datos reales, no llaman a ML ni Storage
- admin_headers fixture ya disponible en conftest.py
- Se crean predicciones reales via API para poblar la BD antes de testear stats
"""
 
import io
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

# Helpers

async def _crear_prediccion(client: AsyncClient, headers: dict) -> dict:
    """Crea una predicción de imagen real en BD para poblar datos de stats."""
    import cv2
    import numpy as np
 
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
 
    with patch(
        "src.services.predictions.StorageService.upload_file",
        new_callable=AsyncMock,
        return_value="test/1/fake.jpg",
    ):
        with patch("src.services.predictions.get_predictor") as mock_pred:
            mock_pred.return_value.predict.return_value = ("A", 0.95)
            response = await client.post(
                "/predictions/image",
                headers=headers,
                data={"modo": "IMAGEN_SUBIDA"},
                files={"archivo": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            )
    assert response.status_code == 200
    return response.json()

# Tests - gestion

class TestAdminGestion:
 
    # 1. GET /admin/users — admin ve todos los usuarios
    async def test_get_all_users_admin(
        self,
        client: AsyncClient,
        admin_headers: dict,
        registered_user: dict,
    ):
        response = await client.get("/admin/users", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        assert "total" in body
        assert "page" in body
        assert body["total"] >= 1
        # Todos los items tienen los campos esperados
        for item in body["items"]:
            assert "id" in item
            assert "email" in item
            assert "nombre_usuario" in item
            assert "status" in item
 
    # 2. GET /admin/users — user normal recibe 403
    async def test_get_all_users_user_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        response = await client.get("/admin/users", headers=auth_headers)
        assert response.status_code == 403
 
    # 3. GET /admin/predictions — admin ve todas las predicciones
    async def test_get_all_predictions_admin(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get("/admin/predictions", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        assert body["total"] >= 1
        item = body["items"][0]
        assert "sesion" in item
        assert "resultado" in item
 
    # 4. GET /admin/logs — admin ve logs de actividad
    async def test_get_logs_admin(
        self,
        client: AsyncClient,
        admin_headers: dict,
        registered_user: dict,
    ):
        # El registro del usuario ya genera un log
        response = await client.get("/admin/logs", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        assert "total" in body
        # Verificar estructura de cada log
        for log in body["items"]:
            assert "id" in log
            assert "usuario_id" in log
            assert "accion" in log
            assert "fecha" in log
 
    # 5. Paginación funciona correctamente
    async def test_paginacion_users(
        self,
        client: AsyncClient,
        admin_headers: dict,
        registered_user: dict,
    ):
        response = await client.get(
            "/admin/users?page=1&limit=1", headers=admin_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["page"] == 1
        assert body["limit"] == 1
        assert len(body["items"]) <= 1
 
    # 6. Sin autenticación → 403
    async def test_sin_auth_forbidden(self, client: AsyncClient):
        response = await client.get("/admin/users")
        assert response.status_code in (401, 403)
        
# Tests - Estadisticas

class TestAdminStats:
 
    # 1. GET /admin/stats/gestures — devuelve lista (puede estar vacía)
    async def test_stats_gestures(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get("/admin/stats/gestures", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert body[0]["gesto"] == "A"
        assert "total" in body[0]
        assert "confianza_media" in body[0]
 
    # 2. GET /admin/stats/users — devuelve usuarios más activos
    async def test_stats_users(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get("/admin/stats/users", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "usuario_id" in body[0]
        assert "total_sesiones" in body[0]
 
    # 3. GET /admin/stats/activity?periodo=day
    async def test_stats_activity_day(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get(
            "/admin/stats/activity?periodo=day", headers=admin_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        # La sesión recién creada debe aparecer en las últimas 24h
        assert len(body) >= 1
        assert "intervalo" in body[0]
        assert "total" in body[0]
 
    # 4. GET /admin/stats/activity?periodo=week
    async def test_stats_activity_week(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get(
            "/admin/stats/activity?periodo=week", headers=admin_headers
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)
 
    # 5. GET /admin/stats/activity — periodo inválido → 422
    async def test_stats_activity_periodo_invalido(
        self,
        client: AsyncClient,
        admin_headers: dict,
    ):
        response = await client.get(
            "/admin/stats/activity?periodo=year", headers=admin_headers
        )
        assert response.status_code == 422
 
    # 6. GET /admin/stats/modes — devuelve todos los modos
    async def test_stats_modes(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get("/admin/stats/modes", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        # Deben aparecer todos los modos definidos en ModoSesion
        modos = [item["modo"] for item in body]
        assert "IMAGEN_SUBIDA" in modos
        assert "FOTO_CAPTURADA" in modos
        assert "VIDEO_SUBIDO" in modos
        assert "VIDEO_GRABADO" in modos
        assert "LIVE_SESSION" in modos
 
    # 7. GET /admin/stats/confidence — devuelve confianza por gesto
    async def test_stats_confidence(
        self,
        client: AsyncClient,
        admin_headers: dict,
        auth_headers: dict,
    ):
        await _crear_prediccion(client, auth_headers)
 
        response = await client.get("/admin/stats/confidence", headers=admin_headers)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "gesto" in body[0]
        assert "confianza_media" in body[0]
        assert "total_muestras" in body[0]
 
    # 8. GET /admin/stats/registrations — devuelve registros por día
    async def test_stats_registrations(
        self,
        client: AsyncClient,
        admin_headers: dict,
        registered_user: dict,
    ):
        response = await client.get(
            "/admin/stats/registrations", headers=admin_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        # El usuario registrado en este test debe aparecer
        assert len(body) >= 1
        assert "dia" in body[0]
        assert "total" in body[0]
 
    # 9. Endpoints de stats son solo para admin
    async def test_stats_solo_admin(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        endpoints = [
            "/admin/stats/gestures",
            "/admin/stats/users",
            "/admin/stats/activity",
            "/admin/stats/modes",
            "/admin/stats/confidence",
            "/admin/stats/registrations",
        ]
        for endpoint in endpoints:
            response = await client.get(endpoint, headers=auth_headers)
            assert response.status_code == 403, (
                f"{endpoint} debería devolver 403 para usuario normal"
            )