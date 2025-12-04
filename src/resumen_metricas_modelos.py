import os
import joblib
import pandas as pd
from src.core_textil import MODELS_DIR
METRICAS_FILES = {
    "Random Forest":        "metricas_rf_class_bal.joblib",
    "Regresión Logística":  "metricas_log_class.joblib",
    "SVM":                  "metricas_svm_class.joblib",
    "Red Neuronal (ANN)":   "metricas_ann_class.joblib",
}

def main():
    filas = []

    for nombre, fname in METRICAS_FILES.items():
        ruta = os.path.join(MODELS_DIR, fname)

        if not os.path.exists(ruta):
            print(f"[ADVERTENCIA] No se encontró el archivo de métricas para {nombre}: {ruta}")
            continue

        metricas = joblib.load(ruta)

        filas.append({
            "Modelo":   nombre,
            "Accuracy": round(metricas.get("accuracy", 0.0), 4),
            "Precision": round(metricas.get("precision", 0.0), 4),
            "Recall":   round(metricas.get("recall",   0.0), 4),
            "F1-Score": round(metricas.get("f1",       0.0), 4),
            "ROC_AUC":  round(metricas.get("auc",      0.0), 4),
        })

    if not filas:
        print("No se pudo cargar ninguna métrica. Revisa la carpeta 'models/'.")
        return

    df_res = pd.DataFrame(filas)
    print("\n--- RESULTADOS EN TEST (Split 70/30) ---")
    print(df_res.to_string(index=False))
if __name__ == "__main__":
    main()
