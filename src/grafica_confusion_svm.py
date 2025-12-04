# src/grafica_confusion_svm.py

import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta de salida
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "figuras", "modelos_clasificacion")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "confusion_svm_only.png")

def main():
    # Matriz de confusión del SVM (la misma de tu figura)
    # Filas = True (High, Low, Medium)
    # Columnas = Predicted (High, Low, Medium)
    cm = np.array([
        [4,    0,    6],     # True High
        [10, 1568,  21],     # True Low
        [4,    0,   28]      # True Medium
    ], dtype=int)

    # Normalizada por fila (para los números entre paréntesis)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    class_names = ["High", "Low", "Medium"]

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proporción", rotation=90)

    # Ticks y labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SVM")

    # Anotaciones: entero + proporción
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            frac = cm_norm[i, j]
            text = f"{count}\n({frac:.2f})"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="black"
            )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print(f"Figura guardada en: {OUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
