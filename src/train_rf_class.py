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
    scaler_rf = StandardScaler()
    X_train_scaled = scaler_rf.fit_transform(X_train)
    X_test_scaled = scaler_rf.transform(X_test)
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_scaled, y_train)

    print("\nDistribución train después del balanceo:")
    print(pd.Series(y_train_bal).value_counts())

    print("\nDistribución test (sin tocar):")
    print(pd.Series(y_test).value_counts())

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
    )

    print("\nEntrenando Random Forest balanceado...")
    rf.fit(X_train_bal, y_train_bal)

    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("\nMÉTRICAS RANDOM FOREST (Balanceado sin fuga)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    clases = rf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=clases)
    print("\nMatriz de confusión [filas=verdadero, columnas=predicho] en orden de clases del modelo:")
    print(list(clases))
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

    joblib.dump(rf, os.path.join(MODELS_DIR, "modelo_curva_rf_class_bal.joblib"))
    joblib.dump(scaler_rf, os.path.join(MODELS_DIR, "scaler_X_rf_class_bal.joblib"))
    joblib.dump(metricas, os.path.join(MODELS_DIR, "metricas_rf_class_bal.joblib"))

    print("\nModelo RF guardado correctamente.")
