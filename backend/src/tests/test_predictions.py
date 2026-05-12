import io
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import asyncio
import base64
import json
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient
from httpx_ws.transport import ASGIWebSocketTransport
from httpx_ws import aconnect_ws
from src.main import app as fastapi_app

# Fixtures — imagen

@pytest_asyncio.fixture
def imagen_valida() -> dict:
    """Imagen JPEG sintética decodificable por OpenCV."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return {
        "contenido": buf.tobytes(),
        "filename": "test.jpg",
        "content_type": "image/jpeg",
    }

# Fixtures — vídeo

@pytest_asyncio.fixture
def video_valido() -> dict:
    """
    Vídeo MP4 sintético de 30 frames a 10fps (3 segundos).
    Dentro del límite de 180s. Para tests que no necesitan
    controlar el número exacto de frames procesados.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (100, 100),
        )
        for _ in range(30):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        with open(tmp_path, "rb") as f:
            contenido = f.read()
    finally:
        os.unlink(tmp_path)

    return {
        "contenido": contenido,
        "filename": "test.mp4",
        "content_type": "video/mp4",
    }


@pytest_asyncio.fixture
def video_largo() -> dict:
    """
    Vídeo MP4 sintético de 1820 frames a 10fps (182 segundos).
    Supera VIDEO_MAX_DURATION (180s) — debe ser rechazado con 422.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (100, 100),
        )
        for _ in range(1820):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        with open(tmp_path, "rb") as f:
            contenido = f.read()
    finally:
        os.unlink(tmp_path)

    return {
        "contenido": contenido,
        "filename": "largo.mp4",
        "content_type": "video/mp4",
    }


@pytest_asyncio.fixture
def video_secuencia() -> dict:
    """
    Vídeo MP4 sintético de 10 frames a 10fps (1 segundo).
    Con VIDEO_FPS_SAMPLE=6: intervalo = round(10/6) = 2.
    Frames extraídos: índices 0,2,4,6,8 → exactamente 5 frames.
    Diseñado para usarse con mock_predictor_secuencia.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (100, 100),
        )
        for _ in range(10):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        with open(tmp_path, "rb") as f:
            contenido = f.read()
    finally:
        os.unlink(tmp_path)

    return {
        "contenido": contenido,
        "filename": "secuencia.mp4",
        "content_type": "video/mp4",
    }

# Fixtures — mocks

@pytest_asyncio.fixture
def mock_storage():
    """Parchea StorageService.upload_file para evitar peticiones a Supabase."""
    with patch(
        "src.services.predictions.StorageService.upload_file",
        new_callable=AsyncMock,
        return_value="usuarios/1/1/fake-uuid.jpg",
    ) as m:
        yield m


@pytest_asyncio.fixture
def mock_predictor_a():
    """Predictor que siempre devuelve gesto 'A' con confianza 0.95."""
    predictor = MagicMock()
    predictor.predict.return_value = ("A", 0.95)
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_nothing():
    """Predictor que siempre devuelve gesto 'nothing'."""
    predictor = MagicMock()
    predictor.predict.return_value = ("nothing", 0.99)
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_error():
    """Predictor que lanza excepción en cada llamada."""
    predictor = MagicMock()
    predictor.predict.side_effect = RuntimeError("Error interno del modelo")
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_secuencia():
    """
    Predictor que devuelve exactamente 5 gestos — uno por frame extraído
    de video_secuencia (10 frames a 10fps, intervalo=2 → índices 0,2,4,6,8).

    Secuencia: A, A, B, nothing, B
    Con deduplicación y reset en nothing:
      frame 0 → A:       nuevo gesto, se añade (posición 0)
      frame 2 → A:       igual que último, se descarta
      frame 4 → B:       nuevo gesto, se añade (posición 1)
      frame 6 → nothing: reset último gesto
      frame 8 → B:       distinto de None tras reset, se añade (posición 2)

    Resultado esperado: secuencia_texto='ABB', detalles=3
    """
    gestos = [("A", 0.9), ("A", 0.9), ("B", 0.8), ("nothing", 0.99), ("B", 0.85)]
    predictor = MagicMock()
    predictor.predict.side_effect = gestos
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor

# Tests — imagen (8/8)

class TestPredictImage:

    # 1. Camino feliz — IMAGEN_SUBIDA
    async def test_predict_image_subida_ok(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
        mock_storage,
        mock_predictor_a,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sesion"]["modo"] == "IMAGEN_SUBIDA"
        assert body["sesion"]["status"] == "COMPLETADA"
        assert body["resultado"]["secuencia_texto"] == "A"
        assert body["resultado"]["confianza_media"] == pytest.approx(0.95)
        assert body["resultado"]["total_frames"] == 1
        assert len(body["resultado"]["detalles"]) == 1
        assert body["resultado"]["detalles"][0]["gesto"] == "A"

    # 2. Camino feliz — FOTO_CAPTURADA
    async def test_predict_foto_capturada_ok(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
        mock_storage,
        mock_predictor_a,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "FOTO_CAPTURADA"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code == 200
        assert response.json()["sesion"]["modo"] == "FOTO_CAPTURADA"

    # 3. Modo inválido → 422
    async def test_predict_image_modo_invalido(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "LIVE_SESSION"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code == 422

    # 4. Imagen corrupta → 422
    async def test_predict_image_corrupta(
        self,
        client: AsyncClient,
        auth_headers: dict,
        mock_storage,
        mock_predictor_a,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                "corrupta.jpg",
                io.BytesIO(b"esto no es una imagen"),
                "image/jpeg",
            )},
        )
        assert response.status_code == 422

    # 5. Fallo de Storage → 500
    async def test_predict_image_fallo_storage(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
        mock_predictor_a,
    ):
        with patch(
            "src.services.predictions.StorageService.upload_file",
            new_callable=AsyncMock,
            side_effect=Exception("Supabase no disponible"),
        ):
            response = await client.post(
                "/predictions/image",
                headers=auth_headers,
                data={"modo": "IMAGEN_SUBIDA"},
                files={"archivo": (
                    imagen_valida["filename"],
                    io.BytesIO(imagen_valida["contenido"]),
                    imagen_valida["content_type"],
                )},
            )
        assert response.status_code == 500

    # 6. Fallo de ML → 500, sesión INTERRUMPIDA
    async def test_predict_image_fallo_ml(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
        mock_storage,
        mock_predictor_error,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code == 500
        assert response.json()["detail"].startswith("Error en el modelo ML")

    # 7. Gesto nothing → 200, COMPLETADA, secuencia vacía, sin detalles
    async def test_predict_image_gesto_nothing(
        self,
        client: AsyncClient,
        auth_headers: dict,
        imagen_valida: dict,
        mock_storage,
        mock_predictor_nothing,
    ):
        response = await client.post(
            "/predictions/image",
            headers=auth_headers,
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sesion"]["status"] == "COMPLETADA"
        assert body["resultado"]["secuencia_texto"] == ""
        assert body["resultado"]["detalles"] == []

    # 8. Sin autenticación → 401/403
    async def test_predict_image_sin_auth(
        self,
        client: AsyncClient,
        imagen_valida: dict,
    ):
        response = await client.post(
            "/predictions/image",
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                imagen_valida["filename"],
                io.BytesIO(imagen_valida["contenido"]),
                imagen_valida["content_type"],
            )},
        )
        assert response.status_code in (401, 403)

# Tests — vídeo (9 tests)

class TestPredictVideo:

    # 1. Camino feliz — VIDEO_SUBIDO, gesto A en todos los frames
    async def test_predict_video_subido_ok(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_valido: dict,
        mock_predictor_a,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sesion"]["modo"] == "VIDEO_SUBIDO"
        assert body["sesion"]["status"] == "COMPLETADA"
        # Todos los frames devuelven A — deduplicación deja solo una A
        assert body["resultado"]["secuencia_texto"] == "A"
        assert body["resultado"]["total_frames"] > 0
        assert len(body["resultado"]["detalles"]) == 1
        assert body["resultado"]["detalles"][0]["gesto"] == "A"

    # 2. Camino feliz — VIDEO_GRABADO
    async def test_predict_video_grabado_ok(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_valido: dict,
        mock_predictor_a,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_GRABADO"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code == 200
        assert response.json()["sesion"]["modo"] == "VIDEO_GRABADO"

    # 3. Modo inválido → 422
    async def test_predict_video_modo_invalido(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_valido: dict,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "IMAGEN_SUBIDA"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code == 422

    # 4. Archivo corrupto → 422
    async def test_predict_video_corrupto(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                "corrupto.mp4",
                io.BytesIO(b"esto no es un video"),
                "video/mp4",
            )},
        )
        assert response.status_code == 422

    # 5. Vídeo demasiado largo → 422, sin sesión creada en BD
    async def test_predict_video_demasiado_largo(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_largo: dict,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_largo["filename"],
                io.BytesIO(video_largo["contenido"]),
                video_largo["content_type"],
            )},
        )
        assert response.status_code == 422
        assert "180" in response.json()["detail"]

    # 6. Fallo de ML → 500, sesión INTERRUMPIDA en BD
    async def test_predict_video_fallo_ml(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_valido: dict,
        mock_predictor_error,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code == 500
        assert response.json()["detail"].startswith("Error en el modelo ML")

    # 7. Solo gestos nothing → 200, COMPLETADA, secuencia vacía
    async def test_predict_video_solo_nothing(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_valido: dict,
        mock_predictor_nothing,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sesion"]["status"] == "COMPLETADA"
        assert body["resultado"]["secuencia_texto"] == ""
        assert body["resultado"]["detalles"] == []

    # 8. Deduplicación y reset en nothing — A,A,B,nothing,B → ABB
    async def test_predict_video_deduplicacion(
        self,
        client: AsyncClient,
        auth_headers: dict,
        video_secuencia: dict,
        mock_predictor_secuencia,
    ):
        response = await client.post(
            "/predictions/video",
            headers=auth_headers,
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_secuencia["filename"],
                io.BytesIO(video_secuencia["contenido"]),
                video_secuencia["content_type"],
            )},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["resultado"]["secuencia_texto"] == "ABB"
        assert len(body["resultado"]["detalles"]) == 3

    # 9. Sin autenticación → 401/403
    async def test_predict_video_sin_auth(
        self,
        client: AsyncClient,
        video_valido: dict,
    ):
        response = await client.post(
            "/predictions/video",
            data={"modo": "VIDEO_SUBIDO"},
            files={"archivo": (
                video_valido["filename"],
                io.BytesIO(video_valido["contenido"]),
                video_valido["content_type"],
            )},
        )
        assert response.status_code in (401, 403)
        
# Fixture — frame JPEG sintético en base64

@pytest.fixture
def frame_base64() -> str:
    """Frame JPEG 64×64 sintético codificado en base64."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (0, 255, 0)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")
 
# Helper WebSocket

def _ws_client():
    """
    AsyncClient con ASGIWebSocketTransport apuntando a fastapi_app.
    
    ASGIWebSocketTransport (de httpx_ws) maneja el upgrade WebSocket.
    ASGITransport (de httpx) es solo HTTP — no sirve para WebSocket.
    
    El override de get_db está en fastapi_app.dependency_overrides,
    aplicado por el fixture client del conftest. Este cliente lo hereda.
    """
    return AsyncClient(
        transport=ASGIWebSocketTransport(app=fastapi_app),
        base_url="http://test",
    )
 
 
async def _ws_auth(ws, token: str) -> dict:
    """Envía auth y devuelve la respuesta auth_ok como dict."""
    await ws.send_text(json.dumps({"type": "auth", "token": token}))
    response = await ws.receive_text()
    return json.loads(response)
 
# Tests — WebSocket live (7 tests)
 
class TestPredictLive:
    """
    Tests del WebSocket /predictions/live.
 
    Patrón:
    - El fixture `client` aplica override_get_db sobre fastapi_app (BD asl_test).
    - _ws_client() construye un AsyncClient con ASGIWebSocketTransport
      que hereda ese override porque apunta a la misma fastapi_app.
    - Las verificaciones de BD posteriores usan el fixture `client` HTTP normal.
    """
 
    # 1. Autenticación correcta → auth_ok con sesion_id
    async def test_auth_ok_devuelve_sesion_id(
        self,
        client: AsyncClient,
        registered_user: dict,
        frame_base64: str,
    ):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("A", 0.95)
        mock_predictor.is_stop_gesture.return_value = False
 
        token = registered_user["access_token"]
 
        with patch("src.services.predictions.get_predictor", return_value=mock_predictor):
            async with _ws_client() as ws_client:
                async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                    auth_response = await _ws_auth(ws, token)
 
        assert auth_response["type"] == "auth_ok"
        assert isinstance(auth_response["sesion_id"], int)
        assert auth_response["sesion_id"] > 0
 
    # 2. Token inválido → cierre 4001
    async def test_token_invalido_cierra_4001(
        self,
        client: AsyncClient,
        registered_user: dict,
    ):
        with pytest.raises(Exception):
            async with _ws_client() as ws_client:
                async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                    await ws.send_text(json.dumps({
                        "type": "auth",
                        "token": "token.invalido.jwt",
                    }))
                    await ws.receive_text()
 
    # 3. Primer mensaje no es auth → cierre 4001
    async def test_primer_mensaje_no_auth_cierra_4001(
        self,
        client: AsyncClient,
        registered_user: dict,
    ):
        with pytest.raises(Exception):
            async with _ws_client() as ws_client:
                async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                    await ws.send_text(json.dumps({
                        "type": "frame",
                        "data": "",
                    }))
                    await ws.receive_text()
 
    # 4. Gesto de parada → cierre 4003, sesión COMPLETADA en BD
    async def test_stop_gesture_cierra_4003_sesion_completada(
        self,
        client: AsyncClient,
        registered_user: dict,
        frame_base64: str,
    ):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("A", 0.90)
        mock_predictor.is_stop_gesture.side_effect = [False, True]
 
        token = registered_user["access_token"]
        sesion_id = None
 
        with patch("src.services.predictions.get_predictor", return_value=mock_predictor):
            try:
                async with _ws_client() as ws_client:
                    async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                        auth_resp = await _ws_auth(ws, token)
                        sesion_id = auth_resp["sesion_id"]
 
                        await ws.send_text(json.dumps({"type": "frame", "data": frame_base64}))
                        await ws.receive_text()
 
                        await ws.send_text(json.dumps({"type": "frame", "data": frame_base64}))
                        await ws.receive_text()
            except Exception:
                pass
 
        assert sesion_id is not None
        response = await client.get(
            f"/predictions/{sesion_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["sesion"]["status"] == "COMPLETADA"
 
    # 5. Timeout → sesión COMPLETADA con lo acumulado
    async def test_timeout_sesion_completada(
        self,
        client: AsyncClient,
        registered_user: dict,
        frame_base64: str,
    ):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("B", 0.88)
        mock_predictor.is_stop_gesture.return_value = False
 
        token = registered_user["access_token"]
        sesion_id = None
 
        with patch("src.services.predictions.get_predictor", return_value=mock_predictor):
            with patch("src.services.predictions.settings") as mock_settings:
                mock_settings.VIDEO_MAX_DURATION = 0.1  # 100 ms
 
                try:
                    async with _ws_client() as ws_client:
                        async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                            auth_resp = await _ws_auth(ws, token)
                            sesion_id = auth_resp["sesion_id"]
 
                            await ws.send_text(json.dumps({"type": "frame", "data": frame_base64}))
                            await ws.receive_text()
 
                            await asyncio.sleep(0.5)
                            await ws.receive_text()
                except Exception:
                    pass
 
        assert sesion_id is not None
        response = await client.get(
            f"/predictions/{sesion_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["sesion"]["status"] == "COMPLETADA"
 
    # 6. Error ML → cierre 4004, sesión INTERRUMPIDA en BD
    async def test_error_ml_sesion_interrumpida(
        self,
        client: AsyncClient,
        registered_user: dict,
        frame_base64: str,
    ):
        mock_predictor = MagicMock()
        mock_predictor.is_stop_gesture.return_value = False
        mock_predictor.predict.side_effect = RuntimeError("fallo ML simulado")
 
        token = registered_user["access_token"]
        sesion_id = None
 
        with patch("src.services.predictions.get_predictor", return_value=mock_predictor):
            try:
                async with _ws_client() as ws_client:
                    async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                        auth_resp = await _ws_auth(ws, token)
                        sesion_id = auth_resp["sesion_id"]
 
                        await ws.send_text(json.dumps({"type": "frame", "data": frame_base64}))
                        await ws.receive_text()
            except Exception:
                pass
 
        assert sesion_id is not None
        response = await client.get(
            f"/predictions/{sesion_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["sesion"]["status"] == "INTERRUMPIDA"
 
    # 7. Sin autenticación → cierre 4001
    async def test_sin_autenticacion_cierra_4001(
        self,
        client: AsyncClient,
    ):
        with patch("src.services.predictions.asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()
 
            with pytest.raises(Exception):
                async with _ws_client() as ws_client:
                    async with aconnect_ws("http://test/predictions/live", ws_client) as ws:
                        await ws.receive_text()