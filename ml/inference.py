""" 
inference.py - ASL Sign Language Translator
Modulo de inferencia: recibe imagen o frame, devuelve (gesto, confianza)

Uso desde el backend:
    from ml.inference import ASLPredictor
    predictor = ASLPredictor(model_path=settings.MODEL_PATH, task_path=settings.TASK_PATH)
    
    # Clasificar gesto
    gesture, confidence = predictor.predict(iaamge_np)
    
    # Detectar gesto de parada (dos palmas abiertas)
    if predictor.is_stop_gesture(frame):
        # cerrar sesion live
"""

import numpy as np
import joblib
import cv2
import mediapipe.tasks as mp_tasks

# CONSTANTES

# Numero de landmarks que devuelve MediaPipe Hand Landmarker
_NUM_LANDMARKS = 21

# Tamaño esperado del vector de features (21 landmarks x 3 coordenadas)
_FEATURE_SIZE = _NUM_LANDMARKS * 3

# Gesto que representa ausencia de mano detectada
_NOTHING_GESTURE = "nothing"

# Confianza asignada cuando no hay mano (no es una prediccion real del modelo)
_NOTHING_CONFIDENCE = 0.0

# Indice del landmark de la muñeca
_WRIST = 0

# Indices de las puntas de cada dedo: pulgar, indice, medio, anular, meñique
_FINGERTIPS = [4, 8, 12, 16, 20]

# Indices de las bases de cada dedo (nudillo proximal)
_FINGER_BASES = [2, 5, 9, 13, 17]

# Minimo de dedos extendidos por mano para considerar palma abierta
_MIN_FINGERS_EXTENDED = 4

# CLASE PRINCIPAL

class ASLPredictor:
    """ 
    Encapsula el detector de landmarks de MediaPipe y el clasificador Random Forest.
    
    Se instancia una sola vez al arrancar el servidor FastAPI y se reutiliza en cada peticion.
    La inicializacion carga desde disco ambos modelos, por lo que es costosa - no crear por peticion
    
    Args:
        model_path: Ruta al archivo .pkl del modelo Random Forest
        task_path: Ruta al archivo .task del MediaPipe Hand Landmarker
        min_detection_confidence: Umbral de confianza para la deteccion de mano.
            Por defecto 0.5. Usar 0.2 para las clases M, N y del
    """
    
    def __init__(
        self,
        model_path: str = "ml/models/asl_model.pkl",
        task_path: str = "ml/hand_landmarker.task",
        min_detection_confidence: float = 0.2,
    ) -> None:
        self._model = self._load_model(model_path)
        self._detector = self._load_detector(task_path, min_detection_confidence)
        
    # API PUBLICA
    
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """ 
        Clasifica el gesto ASL presente en la imagen
        
        Args:
            image: Array numpy en formato BGR (salida estandar de OpenCV)
                    o RGB. Debe ser uint8 con shape (H, W, 3)
            
        Returns:
            Tupla (gesto, confianza) donde:
                - gesto:        String con la letra ASL ("A"-"Z", "del", "space", "nothing")
                - confianza:    Float en [0.0, 1.0]. Es 0.0 si no se detecto mano.
        
        Raises:
            ValueError: si la imagen tiene un formato inesperado
        """
        
        self._validate_image(image)
        
        result = self._detect(image)
        
        # Sin mano detectada
        if not result.hand_landmarks:
            return (_NOTHING_GESTURE, _NOTHING_CONFIDENCE)
        
        features = self._extract_features(result)
        gesture, confidence = self._classify(features)
        return (gesture, confidence)
    
    def is_stop_gesture(self, image: np.ndarray) -> bool:
        """ 
        Detecta el gesto de parada: dos palmas abiertas simultaneas.
        
        Un gesto de parada se considera valido cuando MediaPipe detecta 
        exactamente 2 manos y cada una tiene al menos 4 dedos extendidos.
        
        La extension de cada dedo se mide por distancia euclidiana desde
        la muñeca (landmark 0): un dedo esta extendido si su punta esta
        mas lejos de la muñeca que su base. Este metodo es robusto ante
        distintas orientaciones de la mano (palmas hacia camara, hacia
        arriba, etc.).
        
        Args:
            image: Array numpy en formato BGR, uint8, shape (H, W, 3).
        
        Returns:
            True si se detectan dos palmas abiertas, False en cualquier
            otro caso (una mano, ninguna, manos cerradas).
        
        Raises:
            ValueError: Si la imagen tiene un formato inesperado.
        """
        self._validate_image_image(image)
        
        result = self._detect(image)
        
        # Necesitamos exactamente 2 manos
        if len(result.hand_landmarks) != 2:
            return False
        
        # Ambas manos deben tener al menos _MIN_FINGERS_EXTENDED dedos extendidos
        return all(
            self._is_open_hand(hand)
            for hand in result.hand_landmarks
        )
    
    # INICIALIZACION
    
    @staticmethod
    def _load_model(model_path: str):
        """Carga el modelo Random Forest desde disco"""
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Modelo no encontrado en '{model_path}'"
                "Genera el modelo ejecutando ml/scripts/train.py antes de arrancar el servidor"
            )
        return model
    
    @staticmethod
    def _load_detector(task_path: str, min_detection_confidence: float):
        """
        Inicializa el Hand Landmarker de MediaPipe.
        Configura num_hands=2 para soportar tanto predict() como is_stop_gesture().
        """
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path = task_path)
            options = mp_tasks.vision.HandLandmarkerOptions(
                base_options = base_options,
                num_hands = 2,
                min_hand_detection_confidence = min_detection_confidence,
                min_hand_presence_confidence = min_detection_confidence,
                min_tracking_confidence = min_detection_confidence,
            )
            detector = mp_tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo inicializar MediaPipe Hand Landmarker desde '{task_path}': {exc}"
            )
        return detector
    
    # PIPELINE INTERNO
    
    def _detect(self, image: np.ndarray):
        """ 
        Ejecuta MediaPipe sobre la imagen y devuelve el resultado raw.
        Centraliza la conversion BGR-RGB y la creacion del objeto mp_image.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mp_image = mp_tasks.vision.Image(
            image_format = mp_tasks.vision.ImageFormat.SRGB,
            data = rgb,
        )
        
        result = self._detector.detect(mp_image)
        
    def _extract_features(result) -> np.ndarray:
        """ 
        Extrae el vector de 63 features de la primera mano detectada.
        
        Args:
            result: Resultado de MediaPipe que contiene al menos una mano.
            
        Returns.
            Array numpy de shape (63,).
        """
        
        landmarks = result.hand_landmarks[0] 
        features = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype = np.float32,
        ).flatten()
        
        assert features.shape == (_FEATURE_SIZE,), (
            f"Vector de features inesperado: {features.shape}, esperado ({_FEATURE_SIZE},)"
        )
        return features
    
    def _classify(self, features: np.ndarray) -> tuple[str, float]:
        """ 
        Clasifica el vector de features con el modelo Random Forest
        
        Returns:
            Tupla (gesto, confianza) con la clase predicha y su probabilidad
        """
        features_2d = features.reshape(1, -1)
        
        gesture = self._model.predict(features_2d)[0]
        
        probabilities = self._model.predict_proba(features_2d)[0]
        class_index = list(self._model.classes_).index(gesture)
        confidence = float(probabilities[class_index])
        
        return (gesture, confidence)
    
    @staticmethod
    def _is_open_hand(landmarks) -> bool:
        """ 
        Determina si una mano esta abierta (palma extendida).
        
        Criterio: al menos _MIN_FINGERS_EXTENDED dedos extendidos.
        Un dedo está extendido si la distancia euclidiana de su punta
        a la muñeca es mayor que la distancia de su base a la muñeca.
        
        Args:
            landmarks: Lista de 21 NormalizedLandmark de MediaPipe.
            
        Returns:
            True si la mano esta abierta, False si esta cerrada.
        """
        wrist = np.array([
            landmarks[_WRIST].x,
            landmarks[_WRIST].y,
            landmarks[_WRIST].z,
        ])
        
        extended_count = 0
        
        for tip_idx, base_idx, in zip(_FINGERTIPS, _FINGER_BASES):
            tip = np.array([
                landmarks[tip_idx].x,
                landmarks[tip_idx].y,
                landmarks[tip_idx].z,
            ])
            base = np.array([
                landmarks[base_idx].x,
                landmarks[base_idx].y,
                landmarks[base_idx].z,
            ])
            
            dist_tip = np.linalg.norm(tip - wrist)
            dist_base = np.linalg.norm(base - wrist)
            
            if dist_tip > dist_base:
                extended_count += 1
        
        return extended_count >= _MIN_FINGERS_EXTENDED
    
    # VALIDACION
    
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Valida que la imagen tiene el formato esperado"""
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Se esperaba numpy.ndarray, se recibio {type(image).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"La imagen debe tener shape (H, W, C), se recibio {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"La imagen debe ser uint8, se recibio {image.dtype}"
            )