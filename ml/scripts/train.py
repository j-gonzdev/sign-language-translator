"""
train.py
Entrenamiento del modelo Random Forest para clasificacion ASL.

Entrada:    ml/data/processed/landmarks_train.csv
Salida:     ml/models/asl_model.pkl

Uso:
    python ml/scripts/train.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

# CONFIGURACION

RANDOM_SEED = 42

PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_leaf': [1, 2]
}

CV_FOLDS = 5

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
TRAIN_CSV = ML_DIR / "data" / "processed" / "landmarks_train.csv"
MODELS_DIR = ML_DIR / "models"
MODEL_PATH = MODELS_DIR / "asl_model.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers = [
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
    
    # Modelo base
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # GridSearchCV
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    log.info(f"\nIniciando GridSearchCV...")
    log.info(f"Grid: {PARAM_GRID}")
    log.info(f"Folds")
    log.info(f"Combinaciones: {CV_FOLDS * len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf'])} entrenamientos")
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(x, y)
    
    # Resultados
    log.info(f"\nMejores hiperparametros: {grid_search.best_params_}")
    log.info(f"Mejor accuracy en CV: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Resultados de todas las combinaciones
    log.info("\nResultados por combinacion:")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cols = ['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf',
            'mean_test_score', 'std_test_score', 'rank_test_score']
    log.info("\n" + cv_results[cols].sort_values('rank_test_score').to_string(index=False))
    
    # Guardar modelo
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    log.info(f"nModelo guardado en: {MODEL_PATH}")
    log.info("=" * 60)
    log.info("ENTRENAMIENTO COMPLETADO")
    log.info("=" * 60)
    
if __name__ == "__main__":
    main()