import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_CSV = os.path.join(BASE_DIR, "data", "processed_topitop.csv")
OUT_DIR = os.path.join(BASE_DIR, "figuras", "eda_topitop")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 120
sns.set(style="whitegrid")

print(f"Cargando data limpia: {RUTA_CSV}")
if not os.path.exists(RUTA_CSV):
    raise FileNotFoundError(
        f"No se encontró processed_topitop.csv en {RUTA_CSV}. "
        "Ejecuta preprocessing.py antes de generar el EDA."
    )

df = pd.read_csv(RUTA_CSV)
cols_requeridas = ["cantidad", "minutaje", "min_trab"]
for c in cols_requeridas:
    if c not in df.columns:
        raise ValueError(f"Falta la columna requerida '{c}' en processed_topitop.csv")

if "minutos_producidos" not in df.columns:
    df["minutos_producidos"] = df["minutaje"] * df["cantidad"]

if "eficiencia_pct" not in df.columns and "eficiencia" in df.columns:
    df["eficiencia_pct"] = df["eficiencia"] * 100.0

if "eficiencia_pct" not in df.columns:
    raise ValueError("No se encontró 'eficiencia_pct' ni 'eficiencia' en el dataset limpio.")

for c in ["cantidad", "minutaje", "min_trab", "minutos_producidos", "eficiencia_pct"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["cantidad", "minutaje", "min_trab", "minutos_producidos", "eficiencia_pct"])
df = df[
    (df["cantidad"] > 0)
    & (df["minutaje"] > 0)
    & (df["min_trab"] > 0)
    & (df["eficiencia_pct"] >= 0)
]

labels = ["Baja", "Media", "Alta"]
bins = [0, 70, 85, 120]

if "categoria" not in df.columns:
    df["categoria"] = pd.cut(
        df["eficiencia_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
else:
    df["categoria"] = pd.Categorical(df["categoria"], categories=labels, ordered=True)

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    "Topitop: Exploratory production analysis (EDA panel)\n"
    "Clean data from processed_topitop.csv",
    fontsize=14,
    y=0.98
)

# 1. Conteo por categoría
ax1 = plt.subplot(2, 3, 1)
cont = df["categoria"].value_counts().reindex(labels)
sns.barplot(x=cont.index, y=cont.values, ax=ax1, palette="Set2")
for i, v in enumerate(cont.values):
    ax1.text(i, v + max(cont.values) * 0.02, f"{int(v)}", ha="center", fontsize=9)
ax1.set_title("1) Records per category", fontsize=11)
ax1.set_xlabel("Categoría")
ax1.set_ylabel("Number of records")

# 2. Boxplot Minutaje por categoría
ax2 = plt.subplot(2, 3, 2)
sns.boxplot(x="categoria", y="minutaje", data=df, ax=ax2, palette="Set3")
ax2.set_title("2) Minutaje by category", fontsize=11)
ax2.set_xlabel("Categoría")
ax2.set_ylabel("Minutaje estilo (min/prenda)")

# 3. Boxplot Eficiencia por categoría
ax3 = plt.subplot(2, 3, 3)
sns.boxplot(x="categoria", y="eficiencia_pct", data=df, ax=ax3, palette="Pastel1")
ax3.set_title("3) Eficiencia by category", fontsize=11)
ax3.set_xlabel("Categoría")
ax3.set_ylabel("Eficiencia (%)")

# 4. Histogramas superpuestos de CANTIDAD por categoría
ax4 = plt.subplot(2, 3, 4)
for cat, col in zip(labels, sns.color_palette("husl", 3)):
    subset = df[df["categoria"] == cat]["cantidad"]
    if subset.empty:
        continue
    sns.histplot(
        subset,
        bins=25,
        kde=False,
        stat="count",
        ax=ax4,
        label=str(cat),
        color=col,
        alpha=0.45
    )
ax4.set_title("4) Quantity distribution (overlapped)", fontsize=11)
ax4.set_xlabel("Cantidad")
ax4.set_ylabel("Frequency")
ax4.legend()

# 5. Dispersión Minutos Producidos vs Minutos Permanencia
ax5 = plt.subplot(2, 3, 5)
sns.scatterplot(
    x="minutos_producidos",
    y="min_trab",
    hue="categoria",
    data=df,
    ax=ax5,
    palette="Dark2",
    s=25,
    alpha=0.8
)
ax5.set_title("5) Minutos produced vs Minutes stay", fontsize=11)
ax5.set_xlabel("Minutes produced (Total)")
ax5.set_ylabel("Minutes stay (Min Trab)")
ax5.legend(title="Categoría", loc="lower right", fontsize=8)

# 6. Mapa de calor de correlaciones
ax6 = plt.subplot(2, 3, 6)
num_cols = ["cantidad", "minutaje", "minutos_producidos", "min_trab", "eficiencia_pct"]
corr = df[num_cols].corr()
sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    annot=True,
    cmap="RdYlGn",
    ax=ax6,
    cbar_kws={"shrink": 0.75},
    fmt=".2f"
)
ax6.set_title("6) Correlation between numerical variables", fontsize=11)
ax6.tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.97])

panel_path = os.path.join(OUT_DIR, "panel_eda_topitop.png")
plt.savefig(panel_path, dpi=300)
plt.close(fig)

# -----------------------
# Figuras individuales
# -----------------------
def save_single(fig_fn, draw_func):
    plt.figure(figsize=(6, 4))
    draw_func()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fig_fn), dpi=300)
    plt.close()

# 1) Conteo
def _g1():
    cont = df["categoria"].value_counts().reindex(labels)
    sns.barplot(x=cont.index, y=cont.values, palette="Set2")
    plt.xlabel("Categoría")
    plt.ylabel("Number of records")
    plt.title("Records per category")

save_single("g1_conteo_categoria.png", _g1)

# 2) Box minutaje
save_single(
    "g2_box_minutaje_categoria.png",
    lambda: (
        sns.boxplot(x="categoria", y="minutaje", data=df, palette="Set3"),
        plt.xlabel("Categoría"),
        plt.ylabel("Minutaje estilo (min/prenda)"),
        plt.title("Minutaje by category"),
    )
)

# 3) Box eficiencia
save_single(
    "g3_box_eficiencia_categoria.png",
    lambda: (
        sns.boxplot(x="categoria", y="eficiencia_pct", data=df, palette="Pastel1"),
        plt.xlabel("Categoría"),
        plt.ylabel("Eficiencia (%)"),
        plt.title("Eficiencia by category"),
    )
)

# 4) Hist cantidad
def _hist():
    for cat, col in zip(labels, sns.color_palette("husl", 3)):
        subset = df[df["categoria"] == cat]["cantidad"]
        if subset.empty:
            continue
        sns.histplot(
            subset,
            bins=25,
            kde=False,
            stat="count",
            label=str(cat),
            color=col,
            alpha=0.45
        )
    plt.xlabel("Cantidad")
    plt.ylabel("Frequency")
    plt.title("Quantity distribution by category")
    plt.legend()

save_single("g4_hist_cantidad_superpuesto.png", _hist)

# 5) Scatter minutos producidos vs min_trab
save_single(
    "g5_scatter_totmin_vs_mintrab.png",
    lambda: (
        sns.scatterplot(
            x="minutos_producidos",
            y="min_trab",
            hue="categoria",
            data=df,
            palette="Dark2",
            s=25,
            alpha=0.8
        ),
        plt.xlabel("Minutes produced (Total)"),
        plt.ylabel("Minutes stay (Min Trab)"),
        plt.title("Minutes produced vs Minutes stay"),
    )
)

# 6) Heatmap correlación
save_single(
    "g6_heatmap_correlacion.png",
    lambda: (
        sns.heatmap(
            df[num_cols].corr(),
            vmin=-1,
            vmax=1,
            annot=True,
            cmap="RdYlGn",
            cbar_kws={"shrink": 0.8},
            fmt=".2f"
        ),
        plt.title("Correlation between numerical variables"),
    )
)

print(f"\nListo. Panel guardado en: {panel_path}")
print(f"Carpeta con figuras individuales: {OUT_DIR}")
