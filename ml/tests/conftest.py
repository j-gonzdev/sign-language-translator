"""
conftest.py — Fixtures compartidos para los tests de ML.

Pytest carga este archivo automáticamente. Los fixtures definidos aquí
están disponibles en test_inference.py y test_landmarks.py sin importación.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path

from ml.inference import ASLPredictor

# Rutas base — absolutas para que funcionen independientemente de desde
# dónde se ejecute pytest

# ml/tests/ -> ml/ -> raíz del proyecto
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixture principal — instancia compartida por toda la sesión

@pytest.fixture(scope="session")
def predictor() -> ASLPredictor:
    """
    Instancia de ASLPredictor compartida por todos los tests.

    scope="session" garantiza que MediaPipe y el modelo Random Forest
    se cargan una sola vez por ejecución de pytest, no una vez por test.

    Usa rutas absolutas para que funcione independientemente del directorio
    desde el que se ejecute pytest.
    """
    return ASLPredictor(
        model_path=str(_PROJECT_ROOT / "ml" / "models" / "asl_model.pkl"),
        task_path=str(_PROJECT_ROOT / "ml" / "hand_landmarker.task"),
    )

# Fixtures de imagen — una por caso de uso

@pytest.fixture(scope="session")
def img_hand_a() -> np.ndarray:
    """Imagen BGR de la letra A en ASL. Debe tener mano visible."""
    return _load_image(_FIXTURES_DIR / "hand_A.jpg")


@pytest.fixture(scope="session")
def img_nothing() -> np.ndarray:
    """Imagen BGR sin mano. MediaPipe no debe detectar landmarks."""
    return _load_image(_FIXTURES_DIR / "hand_nothing.jpg")


@pytest.fixture(scope="session")
def img_two_palms() -> np.ndarray:
    """Imagen BGR con dos palmas abiertas. Gesto de parada."""
    return _load_image(_FIXTURES_DIR / "hand_two_palms.jpg")

# Helpers privados

def _load_image(path: Path) -> np.ndarray:
    """
    Carga una imagen desde disco en formato BGR (numpy uint8).
    Falla con mensaje claro si el fixture no existe.
    """
    if not path.exists():
        pytest.fail(
            f"Fixture no encontrado: '{path}'. "
            "Copia la imagen correspondiente en ml/tests/fixtures/ antes de ejecutar los tests."
        )
    image = cv2.imread(str(path))
    if image is None:
        pytest.fail(f"cv2.imread no pudo leer la imagen: '{path}'.")
    return image