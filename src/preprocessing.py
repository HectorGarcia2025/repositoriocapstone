# src/graficos_modelos_class.py
import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# =====================================================
# RUTAS BÁSICAS
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed_topitop.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figuras", "modelos_clasificacion")

os.makedirs(FIG_DIR, exist_ok=True)

# Archivos ya entrenados
MODELOS = {
    "Random Forest": {
        "model_file": "modelo_curva_rf_class_bal.joblib",
        "scaler_file": "scaler_X_rf_class_bal.joblib",
        "tag": "rf",
    },
    "Regresión Logística": {
        "model_file": "modelo_curva_log_class.joblib",
        "scaler_file": "scaler_X_log_class.joblib",
        "tag": "log",
    },
    "SVM": {
        "model_file": "modelo_curva_svm_class.joblib",
        "scaler_file": "scaler_X_svm_class.joblib",
        "tag": "svm",
    },
    "Red Neuronal (ANN)": {
        "model_file": "modelo_curva_ann_class.joblib",
        "scaler_file": "scaler_X_ann_class.joblib",
        "tag": "ann",
    },
}


# =====================================================
# CARGA DE DATOS
# =====================================================
def cargar_dataset():
    if not os.path.exists(DATA_CLEAN_PATH):
        raise FileNotFoundError(f"No se encontró {DATA_CLEAN_PATH}")

    df = pd.read_csv(DATA_CLEAN_PATH)
    df = df.dropna(subset=["categoria"])

    X = df[["cantidad", "minutaje", "min_trab"]].values
    y = df["categoria"].astype(str).values  # 'Baja', 'Media', 'Alta'

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    clases = np.unique(y)
    return X_train, X_test, y_train, y_test, clases


# =====================================================
# CARGA MODELO + SCALER
# =====================================================
def cargar_modelo_y_scaler(nombre_modelo: str):
    cfg = MODELOS[nombre_modelo]
    model_path = os.path.join(MODELS_DIR, cfg["model_file"])
    scaler_path = os.path.join(MODELS_DIR, cfg["scaler_file"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    modelo = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return modelo, scaler


# =====================================================
# EVALUACIÓN DEL MODELO
# =====================================================
def evaluar_modelo(nombre_modelo, X_train, X_test, y_train, y_test, clases):
    modelo, scaler = cargar_modelo_y_scaler(nombre_modelo)

    if scaler is None:
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    y_pred = modelo.predict(X_test_scaled)

    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test_scaled)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    else:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=clases)

    metricas = {
        "Accuracy": acc,
        "Precisión": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": auc,
    }

    return modelo, metricas, cm, y_pred, y_proba


# =====================================================
# 1) BARRAS DE MÉTRICAS
# =====================================================
def plot_metricas_bar(nombre_modelo, metricas):
    etiquetas = list(metricas.keys())
    valores = [metricas[k] for k in etiquetas]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = ax.bar(etiquetas, valores)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title(f"Métricas de evaluación - {nombre_modelo}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"{MODELOS[nombre_modelo]['tag']}_metricas_bar.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =====================================================
# 2) MATRIZ DE CONFUSIÓN
# =====================================================
def plot_confusion(nombre_modelo, cm, clases):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(clases)))
    ax.set_yticks(np.arange(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)

    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {nombre_modelo}")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"{MODELOS[nombre_modelo]['tag']}_matriz_confusion.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =====================================================
# 3) CURVA ROC (micro-average)
# =====================================================
def plot_roc(nombre_modelo, y_test, y_proba, clases):
    if y_proba is None:
        return

    y_bin = label_binarize(y_test, classes=clases)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = roc_auc_score(y_bin, y_proba, multi_class="ovr")

    fig, ax = plt.subplots(figsize=(4, 3))  # tamaño pequeño

    ax.plot(fpr_micro, tpr_micro, linewidth=2, label=f"AUC = {auc_micro:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Curva ROC - {nombre_modelo}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"{MODELOS[nombre_modelo]['tag']}_roc.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =====================================================
# 3b) CURVA PRECISION-RECALL (micro-average)
# =====================================================
def plot_precision_recall(nombre_modelo, y_test, y_proba, clases):
    if y_proba is None:
        return

    y_bin = label_binarize(y_test, classes=clases)
    precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
    ap_micro = average_precision_score(y_bin, y_proba, average="micro")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, linewidth=2, label=f"AP micro = {ap_micro:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall - {nombre_modelo}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"{MODELOS[nombre_modelo]['tag']}_precision_recall.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =====================================================
# 4) CURVAS DE APRENDIZAJE
# =====================================================
def plot_learning_curves(nombre_modelo, modelo_base, X, y):
    clf = clone(modelo_base)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        shuffle=True,
        random_state=42,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    train_loss = 1.0 - train_mean
    val_loss = 1.0 - val_mean

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax1, ax2 = axes

    # Loss
    ax1.plot(train_sizes, train_loss, marker="o", label="Training loss")
    ax1.plot(train_sizes, val_loss, marker="s", label="Validation loss")
    ax1.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax1.set_ylabel("Error (1 - Accuracy)")
    ax1.set_title("Training and validation loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # Accuracy
    ax2.plot(train_sizes, train_mean, marker="o", label="Training accuracy")
    ax2.plot(train_sizes, val_mean, marker="s", label="Validation accuracy")
    ax2.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.90, 1.01)  # zoom arriba para notar diferencias
    ax2.set_title("Training and validation accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Curvas de aprendizaje - {nombre_modelo}", y=1.05, fontsize=12)
    fig.tight_layout()

    fname = os.path.join(FIG_DIR, f"{MODELOS[nombre_modelo]['tag']}_learning_curves.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# 5) COMPARATIVA GLOBAL DE MÉTRICAS
# =====================================================
def plot_metricas_comparativa_global(metricas_por_modelo):
    modelos = list(metricas_por_modelo.keys())
    metricas = ["Accuracy", "Precisión", "Recall", "F1-score", "AUC"]

    data = np.array([[metricas_por_modelo[m][k] for k in metricas] for m in modelos])

    x = np.arange(len(metricas))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, nombre in enumerate(modelos):
        ax.bar(x + i * width, data[i], width, label=nombre)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metricas)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación de métricas entre modelos de clasificación")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    for i in range(len(modelos)):
        for j in range(len(metricas)):
            val = data[i, j]
            ax.text(
                x[j] + i * width,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "comparativa_metricas_4_modelos.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =====================================================
# MAIN
# =====================================================
def main():
    X_train, X_test, y_train, y_test, clases = cargar_dataset()
    metricas_globales = {}

    for nombre_modelo in MODELOS.keys():
        print(f"\n=== Generando gráficas para: {nombre_modelo} ===")

        modelo, metricas, cm, y_pred, y_proba = evaluar_modelo(
            nombre_modelo, X_train, X_test, y_train, y_test, clases
        )

        metricas_globales[nombre_modelo] = metricas

        plot_metricas_bar(nombre_modelo, metricas)
        plot_confusion(nombre_modelo, cm, clases)
        plot_roc(nombre_modelo, y_test, y_proba, clases)
        plot_precision_recall(nombre_modelo, y_test, y_proba, clases)
        plot_learning_curves(nombre_modelo, modelo, X_train, y_train)

    plot_metricas_comparativa_global(metricas_globales)
    print("\n✓ Gráficas generadas en:", FIG_DIR)


if __name__ == "__main__":
    main()
