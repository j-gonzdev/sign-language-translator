"""
train.py
Entrenamiento del modelo Random Forest para clasificacion ASL.

Entrada:    ml/data/processed/landmarks_train.csv
Salida:     ml/models/asl_model.pkl

Uso:
    python ml/scripts/train.py

Nota sobre hiperparametros:
    El GridSearchCV original determino que los mejores hiperparametros son:
        n_estimators=200, max_depth=None, min_samples_leaf=1
    con accuracy CV de 97.70% y accuracy test de 98.20%.

    Para reducir el tamaño del modelo de 622MB a ~155MB y hacerlo viable
    en produccion con 512MB de RAM (Railway Starter), se fija n_estimators=50.
    La perdida de accuracy estimada es ~1% (de 98.20% a ~97.2%).
    max_depth=None se mantiene para preservar la capacidad del modelo en
    clases dificiles como M, N y del.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# CONFIGURACION

RANDOM_SEED = 42

# Hiperparametros fijados tras GridSearchCV previo
N_ESTIMATORS = 50
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
TRAIN_CSV = ML_DIR / "data" / "processed" / "landmarks_train.csv"
MODELS_DIR = ML_DIR / "models"
MODEL_PATH = MODELS_DIR / "asl_model.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ML_DIR / "data" / "processed" / "train.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


# PIPELINE PRINCIPAL

def main():
    log.info("=" * 60)
    log.info("Entrenamiento - ASL RANDOM FOREST")
    log.info("=" * 60)
    log.info(f"Hiperparametros: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}")

    # Verificaciones
    if not TRAIN_CSV.exists():
        log.error(f"No se encuentra: {TRAIN_CSV}")
        sys.exit(1)

    # Cargar datos
    log.info("Cargando landmarks_train.csv...")
    df = pd.read_csv(TRAIN_CSV)
    log.info(f"Shape: {df.shape}")

    x = df.drop(columns=['label']).values
    y = df['label'].values

    log.info(f"Features: {x.shape[1]}")
    log.info(f"Clases: {sorted(np.unique(y))}")
    log.info(f"Distribucion:\n{pd.Series(y).value_counts().sort_index().to_string()}")

    # Entrenar modelo
    log.info("\nEntrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x, y)

    # Evaluacion rapida sobre train (referencia, no metrica real)
    y_pred_train = model.predict(x)
    train_accuracy = (y_pred_train == y).mean()
    log.info(f"Accuracy en train: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    log.info("(Ejecuta evaluate.py sobre el conjunto de test para la metrica real)")

    # Guardar modelo
    joblib.dump(model, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    log.info(f"\nModelo guardado en: {MODEL_PATH} ({size_mb:.1f} MB)")
    log.info("=" * 60)
    log.info("ENTRENAMIENTO COMPLETADO")
    log.info("=" * 60)


if __name__ == "__main__":
    main()