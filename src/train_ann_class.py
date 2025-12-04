import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
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

    scaler_ann = StandardScaler()
    X_train_scaled = scaler_ann.fit_transform(X_train)
    X_test_scaled = scaler_ann.transform(X_test)
    clases_unicas = np.unique(y_train)
    pesos = compute_class_weight(
        class_weight="balanced",
        classes=clases_unicas,
        y=y_train,
    )
    class_weight_dict = {cls: w for cls, w in zip(clases_unicas, pesos)}

    sample_weight = np.array([class_weight_dict[label] for label in y_train])

    ann_clf = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=200,         
        random_state=42,
    )

    print("\nEntrenando red neuronal...")
    ann_clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    y_pred = ann_clf.predict(X_test_scaled)
    y_proba = ann_clf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("\nMÉTRICAS RED NEURONAL ")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    clases = ann_clf.classes_
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

    joblib.dump(ann_clf, os.path.join(MODELS_DIR, "modelo_curva_ann_class.joblib"))
    joblib.dump(scaler_ann, os.path.join(MODELS_DIR, "scaler_X_ann_class.joblib"))
    joblib.dump(metricas, os.path.join(MODELS_DIR, "metricas_ann_class.joblib"))

    print("\nModelo ANN guardado correctamente.")
