import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ============================================================
# RUTAS BÁSICAS
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed_topitop.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figuras", "modelos_clasificacion")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# CONFIGURACIÓN DE CADA MODELO
# ============================================================
MODELOS = {
    "Random Forest": {
        "short": "rf",
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_rf_class_bal.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_rf_class_bal.joblib"),
        "metricas_path": os.path.join(MODELS_DIR, "metricas_rf_class_bal.joblib"),
    },
    "Regresión Logística": {
        "short": "log",
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_log_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_log_class.joblib"),
        "metricas_path": os.path.join(MODELS_DIR, "metricas_log_class.joblib"),
    },
    "SVM": {
        "short": "svm",
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_svm_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_svm_class.joblib"),
        "metricas_path": os.path.join(MODELS_DIR, "metricas_svm_class.joblib"),
    },
    "Red Neuronal (ANN)": {
        "short": "ann",
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_ann_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_ann_class.joblib"),
        "metricas_path": os.path.join(MODELS_DIR, "metricas_ann_class.joblib"),
    },
}


# ============================================================
# CARGA DE DATASET LIMPIO
# ============================================================
def cargar_dataset_limpio():
    if not os.path.exists(DATA_CLEAN_PATH):
        raise FileNotFoundError(
            f"No se encontró {DATA_CLEAN_PATH}. Ejecuta antes el preprocesamiento "
            "para generar el dataset limpio."
        )

    df = pd.read_csv(DATA_CLEAN_PATH)
    df = df.dropna(subset=["categoria"])
    X = df[["cantidad", "minutaje", "min_trab"]].values
    y = df["categoria"].astype(str).values
    return X, y


# ============================================================
# GRÁFICAS
# ============================================================
def plot_matriz_confusion(y_true, y_pred, labels, title, outfile):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  - Matriz de confusión -> {outfile}")


def plot_learning_curve(
    estimator, X_train, y_train, title, outfile, cv=5, train_sizes=None
):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(5, 4))
    plt.plot(
        train_sizes_abs,
        train_mean,
        "o-",
        label="Train",
    )
    plt.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
    )
    plt.plot(
        train_sizes_abs,
        val_mean,
        "s-",
        label="Val",
    )
    plt.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
    )

    plt.title(title)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  - Curva de aprendizaje -> {outfile}")


def plot_bar_metricas(metricas_dict, title, outfile):
    nombres = list(metricas_dict.keys())
    valores = list(metricas_dict.values())

    plt.figure(figsize=(5, 4))
    x = np.arange(len(nombres))
    plt.bar(x, valores)
    plt.xticks(x, nombres)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Valor")
    plt.title(title)
    for i, v in enumerate(valores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  - Barras de métricas -> {outfile}")


def plot_roc_multiclase(y_true, y_score, clases, title, outfile):
    # y_true y clases son strings; y_score = predict_proba
    y_bin = label_binarize(y_true, classes=clases)
    # ROC micro-promedio (como ejemplo general)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  - Curva ROC -> {outfile}")


# ============================================================
# PROCESO POR MODELO
# ============================================================
def generar_graficos_modelo(nombre, cfg, X, y):
    print(f"\n=== Modelo: {nombre} ===")

    short = cfg["short"]
    model_path = cfg["model_path"]
    scaler_path = cfg["scaler_path"]

    if not os.path.exists(model_path):
        print(f"  * No se encontró el modelo en {model_path}, se omite.")
        return

    modelo = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # mismo split que en el entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)  # por si acaso
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Predicciones y probabilidades
    y_pred = modelo.predict(X_test_scaled)

    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test_scaled)
    else:
        # para SVM sin probability, usar decision_function y normalizar
        scores = modelo.decision_function(X_test_scaled)
        # min-max por fila para llevar a [0,1]
        scores = scores - scores.min(axis=1, keepdims=True)
        denom = scores.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        y_proba = scores / denom

    clases = np.unique(y)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr")

    metricas = {
        "Accuracy": acc,
        "Recall": rec,
        "F1-Score": f1,
        "ROC_AUC": auc_ovr,
    }

    print("  Métricas en test:")
    for k, v in metricas.items():
        print(f"    {k}: {v:.4f}")

    # 1) MATRIZ DE CONFUSIÓN
    outfile_cm = os.path.join(FIG_DIR, f"{short}_confusion.png")
    plot_matriz_confusion(
        y_true=y_test,
        y_pred=y_pred,
        labels=clases,
        title=f"Matriz de confusión - {nombre}",
        outfile=outfile_cm,
    )

    # 2) CURVA DE APRENDIZAJE
    outfile_lc = os.path.join(FIG_DIR, f"{short}_learning_curve.png")
    plot_learning_curve(
        estimator=modelo,
        X_train=X_train,
        y_train=y_train,
        title=f"Curva de aprendizaje - {nombre}",
        outfile=outfile_lc,
    )

    # 3) BARRA DE MÉTRICAS
    outfile_bar = os.path.join(FIG_DIR, f"{short}_metricas_bar.png")
    plot_bar_metricas(
        metricas_dict=metricas,
        title=f"Métricas de desempeño - {nombre}",
        outfile=outfile_bar,
    )

    # 4) CURVA ROC (micro-promedio)
    outfile_roc = os.path.join(FIG_DIR, f"{short}_roc.png")
    plot_roc_multiclase(
        y_true=y_test,
        y_score=y_proba,
        clases=clases,
        title=f"Curva ROC (micro) - {nombre}",
        outfile=outfile_roc,
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Cargando dataset limpio...")
    X, y = cargar_dataset_limpio()
    print(f"Total de registros: {len(y)}")
    print(pd.Series(y).value_counts())

    for nombre, cfg in MODELOS.items():
        generar_graficos_modelo(nombre, cfg, X, y)

    print("\nListo. Se generaron las 4 gráficas por modelo en:")
    print(f"  {FIG_DIR}")
