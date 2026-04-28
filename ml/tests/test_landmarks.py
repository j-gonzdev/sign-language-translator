""" 
test_landmarks.py - Tests de la logica interna de landmarks

Testea la extraccion de features y la deteccion de palma abierta
accediendo a los metodos privados de ASLPredictor.

Acceder a metodos privados en tests es aceptable cuando:
    - La logica interna es suficientemente compleja para merecer tests propios
    - Queremos aislar fallos: saber si el problema esta en MediaPipe
      o en el clasificador Random Forest

Tests cubiertos (seccion 7 del documento):
    - Extraccion landmarks imagen valida
    - Extraccion landmarks imagen sin mano
    - Extraccion landmarks devuelve 21 puntos
    - Deteccion gesto de parada con dos palmas
"""

import pytest
import numpy as np

from ml.inference import ASLPredictor, _FEATURE_SIZE, _NUM_LANDMARKS

# Extraccion de landmarks - _detect() y _extract_features()

class TestLandmarkExtraction:
    """Tests sobre la extraccion de landmarks con MediaPipe"""
    
    def test_detects_landmarks_with_valid_image(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """Con una imagen con mano, MediaPipe debe detectar landmarks."""
        result = predictor._detect(img_hand_a)
        assert len(result.hand_landmarks) > 0
 
    def test_no_landmarks_without_hand(
        self, predictor: ASLPredictor, img_nothing: np.ndarray
    ):
        """Sin mano en la imagen, MediaPipe no debe detectar landmarks."""
        result = predictor._detect(img_nothing)
        assert len(result.hand_landmarks) == 0
 
    def test_extracts_21_landmarks(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """MediaPipe debe devolver exactamente 21 landmarks por mano."""
        result = predictor._detect(img_hand_a)
        assert len(result.hand_landmarks) > 0
        landmarks = result.hand_landmarks[0]
        assert len(landmarks) == _NUM_LANDMARKS
 
    def test_feature_vector_has_63_values(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """El vector de features debe tener 63 valores (21 landmarks × 3 coords)."""
        result = predictor._detect(img_hand_a)
        features = predictor._extract_features(result)
        assert features.shape == (_FEATURE_SIZE,)
 
    def test_feature_vector_is_float32(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """El vector de features debe ser float32."""
        result = predictor._detect(img_hand_a)
        features = predictor._extract_features(result)
        assert features.dtype == np.float32
 
    def test_landmark_coordinates_normalized(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """
        Las coordenadas x e y de los landmarks deben estar normalizadas [0, 1].
        La coordenada z puede salir ligeramente de rango — es profundidad relativa.
        """
        result = predictor._detect(img_hand_a)
        features = predictor._extract_features(result)
 
        # Extraemos x e y: posiciones 0,3,6,... (x) y 1,4,7,... (y)
        x_coords = features[0::3]
        y_coords = features[1::3]
 
        assert x_coords.min() >= 0.0 and x_coords.max() <= 1.0
        assert y_coords.min() >= 0.0 and y_coords.max() <= 1.0
        
# Deteccion de palma abierta - _is_open_hand()

class TestIsOpenHand:
    """Tests sobre la lógica de detección de palma abierta."""
 
    def test_open_hand_detected_with_two_palms(
        self, predictor: ASLPredictor, img_two_palms: np.ndarray
    ):
        """
        Con dos palmas abiertas, _is_open_hand() debe devolver True
        para cada una de las dos manos detectadas.
        """
        result = predictor._detect(img_two_palms)
        assert len(result.hand_landmarks) == 2, (
            "MediaPipe no detectó 2 manos en hand_two_palms.jpg. "
            "Verifica que la imagen tiene ambas manos claramente visibles."
        )
        for hand in result.hand_landmarks:
            assert predictor._is_open_hand(hand) is True
 
    def test_two_palms_triggers_stop_gesture(
        self, predictor: ASLPredictor, img_two_palms: np.ndarray
    ):
        """
        El flujo completo: _detect() → 2 manos → _is_open_hand() × 2 → True.
        Este test verifica la integración de los métodos internos.
        """
        result = predictor._detect(img_two_palms)
        stop = all(
            predictor._is_open_hand(hand)
            for hand in result.hand_landmarks
        )
        assert stop is True
 
    def test_single_hand_does_not_trigger_stop(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """Con una sola mano no debe cumplirse la condición de parada."""
        result = predictor._detect(img_hand_a)
        # Con una sola mano no puede haber gesto de parada (necesitamos 2)
        assert len(result.hand_landmarks) != 2 or not all(
            predictor._is_open_hand(hand)
            for hand in result.hand_landmarks
        )
        
# Robustez del detector

class TestDetectorRobustness:
    """Tests de robustez del detector ante entradas límite."""
 
    def test_detect_returns_empty_landmarks_not_none(
        self, predictor: ASLPredictor, img_nothing: np.ndarray
    ):
        """
        _detect() nunca debe devolver None — MediaPipe siempre devuelve
        un objeto resultado, aunque hand_landmarks esté vacío.
        """
        result = predictor._detect(img_nothing)
        assert result is not None
        assert result.hand_landmarks is not None
 
    def test_detect_with_valid_image_returns_result(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """_detect() con imagen válida debe devolver un resultado con landmarks."""
        result = predictor._detect(img_hand_a)
        assert result is not None
        assert len(result.hand_landmarks) >= 1