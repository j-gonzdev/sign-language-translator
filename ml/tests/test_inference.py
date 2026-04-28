""" 
test_inference.py - Test de la API de ASLPredictor.

Testea predict() e is_stop_gesture() como caja negra
No verifica internos de MediaPipe - solo que el contrato publico se cumple

Test cubiertos (seccion 7 del documento):
    - Inferencia con imagen valida
    - Inferencia con imagen sin mano detectada
    - Inferencia devuelve gesto correcto
    - Inferencia confianza entre 0 y 1
"""

import pytest
import numpy as np

from ml.inference import ASLPredictor

# predict() - contrato de tipos y rangos

class TestPredictContract:
    """predict() debe cumplir su contrato de tipos y rangos siempre"""
    
    def test_returns_tuple(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """predict() debe devolver una tupla."""
        result = predictor.predict(img_hand_a)
        assert isinstance(result, tuple)
 
    def test_returns_two_elements(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """La tupla debe tener exactamente dos elementos."""
        result = predictor.predict(img_hand_a)
        assert len(result) == 2
 
    def test_gesture_is_string(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """El primer elemento debe ser un string."""
        gesture, _ = predictor.predict(img_hand_a)
        assert isinstance(gesture, str)
 
    def test_confidence_is_float(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """El segundo elemento debe ser un float."""
        _, confidence = predictor.predict(img_hand_a)
        assert isinstance(confidence, float)
 
    def test_confidence_between_0_and_1(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """La confianza debe estar en el rango [0.0, 1.0]."""
        _, confidence = predictor.predict(img_hand_a)
        assert 0.0 <= confidence <= 1.0
        
# predict() - caso con mano detectada

class TestPredictWithHand:
    """Comportamiento de predict() cuando hay una mano visible"""
    
    def test_returns_correct_gesture(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """
        Con una imagen clara de la letra A debe predecir 'A'.
        El modelo tiene 98.20% de accuracy — este caso debe ser correcto.
        """
        gesture, _ = predictor.predict(img_hand_a)
        assert gesture == "A"
 
    def test_confidence_above_threshold(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """
        Con una imagen clara la confianza debe ser alta.
        Umbral conservador de 0.5 — si falla, la imagen fixture es ambigua.
        """
        _, confidence = predictor.predict(img_hand_a)
        assert confidence >= 0.5
 
    def test_gesture_is_known_class(self, predictor: ASLPredictor, img_hand_a: np.ndarray):
        """El gesto devuelto debe ser una clase conocida del modelo."""
        known_classes = set(predictor._model.classes_)
        gesture, _ = predictor.predict(img_hand_a)
        assert gesture in known_classes
        
# predict() - caso sin mano detectada

class TestPredictWithoutHand:
    """Comportamiento de predict() cuando no hay manos en la imagen"""
    
    def test_returns_nothing_gesture(self, predictor: ASLPredictor, img_nothing: np.ndarray):
        """Sin mano debe devolver 'nothing'"""
        gesture, _ = predictor.predict(img_nothing)
        assert gesture == "nothing"
        
    def test_returns_zero_confidence(self, predictor: ASLPredictor, img_nothing: np.ndarray):
        """Sin mano la confianza debe ser 0.0 - no hay prediccion real"""
        _, confidence = predictor.predict(img_nothing)
        assert confidence == 0.0
        
# predict() - validacion de entrada

class TestPredictValidation:
    """predict() debe fallar con mensajes claros ante entradas inválidas."""
 
    def test_raises_on_non_array(self, predictor: ASLPredictor):
        """Debe lanzar ValueError si la entrada no es numpy array."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            predictor.predict("no soy una imagen")
 
    def test_raises_on_wrong_shape(self, predictor: ASLPredictor):
        """Debe lanzar ValueError si el array no tiene shape (H, W, 3)."""
        bad_image = np.zeros((100, 100), dtype=np.uint8)  # 2D, sin canal
        with pytest.raises(ValueError, match="shape"):
            predictor.predict(bad_image)
 
    def test_raises_on_wrong_dtype(self, predictor: ASLPredictor):
        """Debe lanzar ValueError si el dtype no es uint8."""
        bad_image = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            predictor.predict(bad_image)
            
# is_stop_gesture()

class TestIsStopGesture:
    """is_stop_gesture() debe detectar dos palmas abiertas."""
 
    def test_returns_true_with_two_palms(
        self, predictor: ASLPredictor, img_two_palms: np.ndarray
    ):
        """Con dos palmas abiertas debe devolver True."""
        assert predictor.is_stop_gesture(img_two_palms) is True
 
    def test_returns_false_with_one_hand(
        self, predictor: ASLPredictor, img_hand_a: np.ndarray
    ):
        """Con una sola mano debe devolver False."""
        assert predictor.is_stop_gesture(img_hand_a) is False
 
    def test_returns_false_with_no_hand(
        self, predictor: ASLPredictor, img_nothing: np.ndarray
    ):
        """Sin manos debe devolver False."""
        assert predictor.is_stop_gesture(img_nothing) is False
 
    def test_returns_bool(self, predictor: ASLPredictor, img_two_palms: np.ndarray):
        """El valor de retorno debe ser bool, no truthy/falsy."""
        result = predictor.is_stop_gesture(img_two_palms)
        assert isinstance(result, bool)