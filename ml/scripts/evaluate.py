""" 
evaluate.py
Evaluacion del modelo Random Foreste entrenado para clasificacion ASL

Entrada: ml/data/processed/landmarks_test.csv
         ml/models/asl_model.pkl
Salida:  ml/data/processed/evaluate.log
         ml/data/processed/confusion_matrix.png
         
Uso:
    python ml/scripts/evaluate.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# CONFIGURACION

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
TEST_CSV = ML_DIR / "data" / "processed" / "landmarks_test.csv"
MODEL_PATH = ML_DIR / "models" / "asl_model.pkl"
PROCESSED_DIR = ML_DIR / "data" / "processed"

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROCESSED_DIR / "evaluate.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# PIPELINE

def main():
    log.info("=" * 60)
    log.info("EVALUACION - ASL RANDOM FOREST")
    log.info("=" * 60)
    
    # Verificaciones
    if not TEST_CSV.exists():
        log.error(f"No se encuentra: {TEST_CSV}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        log.error(f"No se encuentra: {MODEL_PATH}")
        sys.exit(1)
    
    # Cargar datos
    log.info("Cargando landmarks_test.csv...")
    df = pd.read_csv(TEST_CSV)
    log.info(f"Shape: {df.shape}")
    
    x_test = df.drop(columns=['label']).values
    y_test = df['label'].values
    
    # Cargar modelo
    log.info("Cargando modelo...")
    model = joblib.load(MODEL_PATH)
    log.info(f"Modelo: {model}")
    
    # Predicciones
    log.info("Generando predicciones...")
    y_pred = model.predict(x_test)
    
    # Accuracy global
    accuracy = accuracy_score(y_test, y_pred)
    log.info(f"\nAccuracy en test: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    report = classification_report(y_test, y_pred, digits=4)
    log.info(f"\nClassification Report:\n{report}")
    
    # Clases con F1 bajo (umbral 0.90)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    clases_debiles = {
        clase: metrics
        for clase, metrics in report_dict.items()
        if clase not in ['accuracy', 'macro avg', 'weight avg']
        and metrics['f1-score'] < 0.90
    }
    
    if clases_debiles:
        log.info("\nClases con F1 < 0.90:")
        for clase, metrics in sorted(clases_debiles.items(), key=lambda x: x[1]['f1-score']):
            log.info(
                f"  {clase}: F1={metrics['f1-score']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
            )
    else:
        log.info("\nTodas las clases tienen F1 >= 0.90")
        
    # Matriz de confusion
    log.info("\nGenerando matriz de confusion...")
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f'Matriz de Cofusion - Accuracy: {accuracy*100:.2f}%', fontsize=14)
    ax.set_xlabel('Prediccion', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    plt.tight_layout()
    
    cm_path = PROCESSED_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    log.info(f"Guardada en: {cm_path}")
    
    log.info("=" * 60)
    log.info("EVALUACION COMPLETADA")
    log.info("=" * 60)
    
if __name__ == "__main__":
    main()