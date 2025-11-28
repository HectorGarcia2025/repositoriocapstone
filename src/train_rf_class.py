import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed_topitop.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def cargar_dataset_limpio():
    """
    Dataset ya sale del preprocesamiento oficial:

      - Integración L72/L79
      - cantidad, minutaje, min_trab
      - minutos_producidos = minutaje * cantidad
      - eficiencia_pct = (minutos_producidos / min_trab) * 100, acotado [0,120]
      - categoria: Baja[0,70), Media[70,85), Alta[85,100)
    """
    print("Cargando dataset limpio desde:", DATA_CLEAN_PATH)
    if not os.path.exists(DATA_CLEAN_PATH):
        raise FileNotFoundError(
            f"No se encontró {DATA_CLEAN_PATH}. Ejecuta antes el preprocesamiento."
        )
    df = pd.read_csv(DATA_CLEAN_PATH)
    df = df.dropna(subset=["categoria"])
    return df


if __name__ == "__main__":
    df = cargar_dataset_limpio()

    print("\nDistribución original:")
    print(df["categoria"].value_counts())

    X = df[["cantidad", "minutaje", "min_trab"]].values
    y = df["categoria"].astype(str).values 

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    print("\nDistribución train:")
    print(pd.Series(y_train).value_counts())
    print("\nDistribución test:")
    print(pd.Series(y_test).value_counts())

    scaler_rf = StandardScaler()
    X_train_scaled = scaler_rf.fit_transform(X_train)
    X_test_scaled = scaler_rf.transform(X_test)

    print("\nMÉTRICAS RANDOM FOREST")
    rf_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf_clf.fit(X_train_scaled, y_train)
    y_pred = rf_clf.predict(X_test_scaled)
    y_proba = rf_clf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    clases = rf_clf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=clases)
    print("\nClases (ordenadas):", list(clases))
    print("Matriz de confusión [filas=verdadero, columnas=predicho]:")
    print(cm)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    metricas = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }

    joblib.dump(rf_clf, os.path.join(MODELS_DIR, "modelo_curva_rf_class_bal.joblib"))
    joblib.dump(scaler_rf, os.path.join(MODELS_DIR, "scaler_X_rf_class_bal.joblib"))
    joblib.dump(metricas, os.path.join(MODELS_DIR, "metricas_rf_class_bal.joblib"))

    print("\nModelo Random Forest balanceado guardado correctamente.")
