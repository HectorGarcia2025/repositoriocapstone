import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import RandomOverSampler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed_topitop.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def cargar_dataset_limpio():
    print("Cargando dataset limpio desde:", DATA_CLEAN_PATH)
    if not os.path.exists(DATA_CLEAN_PATH):
        raise FileNotFoundError(
            f"No se encontró {DATA_CLEAN_PATH}. Ejecuta antes el preprocesamiento "
            "para generar el dataset limpio."
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

    print("\nDistribución train antes del balanceo:")
    print(pd.Series(y_train).value_counts())

    scaler_log = StandardScaler()
    X_train_scaled = scaler_log.fit_transform(X_train)
    X_test_scaled = scaler_log.transform(X_test)

    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_scaled, y_train)

    print("\nDistribución train después del balanceo:")
    print(pd.Series(y_train_bal).value_counts())

    print("\nDistribución test (sin tocar):")
    print(pd.Series(y_test).value_counts())

    log_clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        class_weight=None,
        random_state=42,
    )

    print("\nEntrenando Regresión Logística balanceada...")
    log_clf.fit(X_train_bal, y_train_bal)

    y_pred = log_clf.predict(X_test_scaled)
    y_proba = log_clf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("\nMÉTRICAS REGRESIÓN LOGÍSTICA)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    clases = log_clf.classes_
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

    joblib.dump(log_clf, os.path.join(MODELS_DIR, "modelo_curva_log_class.joblib"))
    joblib.dump(scaler_log, os.path.join(MODELS_DIR, "scaler_X_log_class.joblib"))
    joblib.dump(metricas, os.path.join(MODELS_DIR, "metricas_log_class.joblib"))

    print("\nModelo de Regresión Logística guardado correctamente.")
