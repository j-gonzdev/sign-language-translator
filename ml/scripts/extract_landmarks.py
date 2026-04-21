"""
extract_landmarks.py
Pipeline de extraccion de landmarks para el dataset ASL.

Entrada:    ml/data/raw/train/ (29 clases x -3200 imagenes)
Salida:     ml/data/processed/landmarks_train.csv
            ml/data/processed/landmarks_test.csv

Uso: python ml/scripts/extract_landmarks.py
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mediapipe as mp
import mediapipe.tasks as mp_tasks

# Mediapipe configuracion
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
RunningMode = mp_tasks.vision.RunningMode
BaseOptions = mp_tasks.BaseOptions

# CONFIGURACION

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Clases con problemas de encuadre - usar umbral bajo
CLASES_UMBRAL_BAJO = {'M', 'N', 'del'}
UMBRAL_ESTANDAR = 0.5
UMBRAL_BAJO = 0.2

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
RAW_TRAIN_DIR = ML_DIR / "data" / "raw" / "train"
PROCESSED_DIR = ML_DIR / "data" / "processed"
MODEL_PATH = ML_DIR / "hand_landmarker.task"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROCESSED_DIR / "extract_landmarks.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# FUNCIONES

def crear_detector(umbral: float):
    options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = str(MODEL_PATH)),
        running_mode = RunningMode.IMAGE,
        num_hands = 1,
        min_hand_detection_confidence = umbral
    )
    return HandLandmarker.create_from_options(options)

def extraer_landmarks(img_path: Path, detector) -> np.ndarray | None:
    """ 
    Extrae el vector de 63 features de una imagen.
    Devuelve np.ndarray de shape (63,) o None si no se detecta mano.
    La clase 'nothing' siempre devuelve vector de ceros.
    """
    clase = img_path.parent.name
    
    # 'nothing' no tienen mano - vector de ceros por definicion
    if clase == 'nothing':
        return np.zeros(63)
    
    try:
        mp_image = mp.Image.create_from_file(str(img_path))
        result = detector.detect(mp_image)
        
        if not result.hand_landmarks:
            return None
        
        landmarks = result.hand_landmarks[0] # primera mano detectada
        vector = []
        for lm in landmarks:
            vector.extend([lm.x, lm.y, lm.z])
            
        return np.array(vector, dtype=np.float32)
    
    except Exception as e:
        log.warning(f"Error procesando {img_path}: {e}")
        return None
    
def recopilar_paths(train_dir: Path) -> list[tuple[Path, str]]:
    """Devuelve lista de (path, clase) para todas las imagenes de train"""
    paths = []
    extensiones = {'.jpg', '.jpeg', '.png'}
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in extensiones:
                    paths.append((img_file, class_dir.name))
    return paths

# PIPELINE PRINCIPAL

def main():
    log.info("="*60)
    log.info("EXTRACCION DE LANDMARKS - ASL DATASET")
    log.info("="*60)
    
    # Verificaciones
    if not RAW_TRAIN_DIR.exists():
        log.error(f"No se encuentra el directorio de train: {RAW_TRAIN_DIR}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        log.error(f"No se encuentra el modelo MediaPipe: {MODEL_PATH}")
        sys.exit(1)
        
    # Recopilar paths
    log.info("Recopilando imagene...")
    all_paths = recopilar_paths(RAW_TRAIN_DIR)
    log.info(f"Total imagenes encontradas: {len(all_paths):,}")
    
    # Crear detectores
    detector_estandar = crear_detector(UMBRAL_ESTANDAR)
    detector_bajo = crear_detector(UMBRAL_BAJO)
    
    # Extraccion
    rows = []
    n_total = len(all_paths)
    n_fallidas = 0
    fallos_por_clase: dict[str, int] = {}
    
    log.info("Extrayendo landmarks...")
    
    for i, (img_path, clase) in enumerate(all_paths):
        if i % 5000 == 0:
            log.info(f"     Progreso: {i:,} / {n_total:,}")
        
        detector = detector_bajo if clase in CLASES_UMBRAL_BAJO else detector_estandar
        vector = extraer_landmarks(img_path, detector)
        
        if vector is None:
            n_fallidas += 1
            fallos_por_clase[clase] = fallos_por_clase.get(clase, 0) + 1
            continue
        
        row = {'label': clase}
        for j, val in enumerate(vector):
                row[f'f{j}'] = val
        rows.append(row)
        
    detector_estandar.close()
    detector_bajo.close()
    
    # Resultados de extraccion
    n_exitosas = len(rows)
    log.info(f"\nExtraccion completada:")
    log.info(f"  Exitosas:  {n_exitosas:,}")
    log.info(f"  Fallidas:  {n_fallidas:,} ({n_fallidas/n_total*100:.1f}%)")
    
    if fallos_por_clase:
        log.info("  Fallos por clase:")
        for clase, n in sorted(fallos_por_clase.items(), key=lambda x: -x[1]):
            log.info(f"     {clase}: {n}")
            
    # Construir DataFrame
    df = pd.DataFrame(rows)
    log.info(f"\nDataFrame: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    log.info(f"Distribucion por clase:\n{df['label'].value_counts().sort_index().to_string()}")
    
    # Division 80/20 estratificada
    log.info("\nAplicando division 80/20 estratificada...")
    df_train, df_test, = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    log.info(f" Train: {len(df_train):,} filas")
    log.info(f" Test: {len(df_test):,} filas")
    
    # Guardar CSVs
    train_path = PROCESSED_DIR / "landmarks_train.csv"
    test_path = PROCESSED_DIR / "landmarks_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    log.info(f"\nGuardado: {train_path}")
    log.info(f"Guardado: {test_path}")
    log.info("="*60)
    log.info("EXTRACCION COMPLETADA")
    log.info("="*60)
    
if __name__ == "__main__":
    main()