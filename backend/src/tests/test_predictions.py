import io
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Fixtures — imagen
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures — vídeo
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
def video_valido() -> dict:
    """
    Vídeo MP4 sintético generado con cv2.VideoWriter.
    30 frames a 10fps = 3 segundos — dentro del límite de 180s.
    Se escribe a disco (VideoWriter no soporta memoria) y se lee como bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,  # fps
            (100, 100),
        )
        for _ in range(30):  # 30 frames → 3 segundos
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            writer.write(frame)
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
    Vídeo MP4 sintético que supera VIDEO_MAX_DURATION (180s).
    1820 frames a 10fps = 182 segundos.
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
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            writer.write(frame)
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


# ---------------------------------------------------------------------------
# Fixtures — mocks compartidos
# ---------------------------------------------------------------------------

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
    """Predictor que devuelve gesto 'A' con confianza 0.95."""
    predictor = MagicMock()
    predictor.predict.return_value = ("A", 0.95)
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_nothing():
    """Predictor que devuelve gesto 'nothing'."""
    predictor = MagicMock()
    predictor.predict.return_value = ("nothing", 0.99)
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_error():
    """Predictor que lanza excepción al predecir."""
    predictor = MagicMock()
    predictor.predict.side_effect = RuntimeError("Error interno del modelo")
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


@pytest_asyncio.fixture
def mock_predictor_secuencia():
    """
    Predictor que devuelve A, A, B, nothing, B en orden.
    Con deduplicación de consecutivos y reset en nothing:
    A, A → A (dedup)
    B → B
    nothing → reset
    B → B (no es consecutivo tras el reset)
    Resultado esperado: 'ABB' → no, espera:
    posición 0: A
    posición 1: B
    nothing: reset ultimo_gesto
    posición 2: B  ← distinto de None, se añade
    secuencia_texto = 'ABB'
    """
    gestos = [("A", 0.9), ("A", 0.9), ("B", 0.8), ("nothing", 0.99), ("B", 0.85)]
    predictor = MagicMock()
    predictor.predict.side_effect = gestos
    with patch("src.services.predictions.get_predictor", return_value=predictor):
        yield predictor


# ---------------------------------------------------------------------------
# Tests — imagen (8/8 existentes, sin cambios)
# ---------------------------------------------------------------------------

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

    # 6. Fallo de ML → 500, sesión INTERRUMPIDA en BD
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

    # 7. Gesto nothing → 200, COMPLETADA, secuencia_texto vacía, sin detalles
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


# ---------------------------------------------------------------------------
# Tests — vídeo
# ---------------------------------------------------------------------------

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

    # 5. Vídeo demasiado largo → 422, sin sesión creada
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
        video_valido: dict,
        mock_predictor_secuencia,
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