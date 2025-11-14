# src/dashboard_textil.py
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime
from pathlib import Path

def _show_image_if_exists(path: str, caption: str = "") -> bool:
    """Renderiza la imagen solo si el archivo existe y es archivo regular."""
    p = Path(path)
    if p.is_file():
        st.image(str(p), caption=caption, use_container_width=True)
        return True
    return False
# ======================================
# CONFIGURACIÓN GENERAL
# ======================================
st.set_page_config(
    page_title="Dashboard Curva de Aprendizaje - Topitop",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH_DEFAULT = os.path.join(BASE_DIR, "data", "2 Salida de prendas.xlsx")

def _find_models_dir():
    candidatos = [
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "modelo_textil_ml", "models"),
        os.path.join(BASE_DIR, "..", "modelo_textil_ml", "models"),
        os.path.join(BASE_DIR, "..", "models"),
    ]
    for c in candidatos:
        if os.path.isdir(c):
            return os.path.abspath(c)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    return os.path.join(BASE_DIR, "models")

MODELS_DIR = _find_models_dir()

# ======================================
# ARCHIVOS DE MODELOS (ajusta si cambian)
# ======================================
MODELOS = {
    "Random Forest (Bal)": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_rf_class_bal.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_rf_class_bal.joblib"),
        "metricas_file": "metricas_rf_class_bal.joblib",
    },
    "Regresión Logística": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_log_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_log_class.joblib"),
        "metricas_file": "metricas_log_class.joblib",
    },
    "SVM": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_svm_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_svm_class.joblib"),
        "metricas_file": "metricas_svm_class.joblib",
    },
    # Si tu ANN la guardaste como .keras, cambia aquí.
    "Red Neuronal (ANN)": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_ann_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_ann_class.joblib"),
        "metricas_file": "metricas_ann_class.joblib",
    },
}

FIG_DIR = os.path.join(BASE_DIR, "figuras")

# ======================================
# ESTILOS
# ======================================
st.markdown("""
<style>
body { background-color:#f7f9fb; color:#2e3b4e; font-family:'Segoe UI',sans-serif; }
.main-header{
  background:linear-gradient(90deg,#d81f26,#222);
  color:#fff; padding:15px 25px; border-radius:8px;
  display:flex; justify-content:space-between; align-items:center;
}
.main-header img{ height:55px; }
.metric-box{
  background-color:#fff; border:2px solid #d81f26; border-radius:10px;
  box-shadow:0 3px 6px rgba(0,0,0,.1); text-align:center; padding:15px; color:#000;
}
.metric-box h3{ color:#d81f26; margin-bottom:6px; }
h2,h3{ color:#d81f26; }
.dataframe{ background-color:#fff !important; border-radius:10px; }
div[data-testid="stAlert"]{
  background-color:#d81f26 !important; border:1px solid #a1151c !important;
  border-radius:8px !important; color:white !important;
}
div[data-testid="stAlert"] *{ color:white !important; }
</style>
""", unsafe_allow_html=True)

# ======================================
# ENCABEZADO
# ======================================
fecha_actual = datetime.now().strftime("%d/%m/%Y")
dataset_nombre = os.path.basename(DATA_PATH_DEFAULT)
st.markdown(f"""
<div class='main-header'>
  <div>
    <h1>Modelo Predictivo de Curva de Aprendizaje Textil</h1>
    <p>Sistema prototipo de apoyo a la decisión - Empresa Topitop</p>
    <small>Actualización: {fecha_actual} | Dataset base: {dataset_nombre}</small>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ======================================
# SIDEBAR (solo navegación)
# ======================================
st.sidebar.image("src/topitop_logo.png", width=150)
st.sidebar.header("Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Resumen general",
        "Sistema predictivo",
        "Comparación de modelos",
        "Curvas de entrenamiento por modelo",
        "Información del proyecto",
    ],
)

# ======================================
# UTILIDADES
# ======================================
def cargar_datos_default():
    hojas = ["L72", "L79"]
    df_list = []
    xls = pd.ExcelFile(DATA_PATH_DEFAULT)
    for h in hojas:
        if h in xls.sheet_names:
            df_list.append(xls.parse(h))
    if not df_list:
        raise RuntimeError("No se hallaron hojas L72/L79 en el Excel base.")
    return pd.concat(df_list, ignore_index=True)

def preprocesar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Plantilla:
      - cantidad ('cantidad' o 'cantidad producida')
      - minutaje estándar por prenda ('mix', 'minutaje', o derivado de 'total min')
      - min_trab (minutos permanencia: 'min trab', 'min_trab', 'permanencia')
    Fórmula:
      minutos_producidos = minutaje * cantidad
      eficiencia_pct     = (minutos_producidos / min_trab) * 100
    """
    df = df_raw.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace("'", "", regex=False)
                  .str.replace('"', "", regex=False)
    )
    if "tipo" in df.columns:
        df = df[df["tipo"].str.contains("salida", case=False, na=False)]

    # cantidad
    cand_cant = [c for c in df.columns if "cant" in c]
    if not cand_cant:
        return pd.DataFrame()
    col_cant = cand_cant[0]

    # min_trab
    cand_trab = [c for c in df.columns if "min trab" in c or "min_trab" in c or "perman" in c]
    if not cand_trab:
        return pd.DataFrame()
    col_min_trab = cand_trab[0]

    # minutaje
    cand_minutaje = [c for c in df.columns if "mix" in c or "minutaje" in c or ("minuto" in c and "prenda" in c)]
    if cand_minutaje:
        col_minutaje = cand_minutaje[0]
        df["minutaje"] = pd.to_numeric(df[col_minutaje], errors="coerce")
    else:
        cand_totalmin = [c for c in df.columns if "total" in c and "min" in c]
        if not cand_totalmin:
            return pd.DataFrame()
        col_total_min = cand_totalmin[0]
        df["minutaje"] = pd.to_numeric(df[col_total_min], errors="coerce") / pd.to_numeric(df[col_cant], errors="coerce")

    df_feat = df[[col_cant, "minutaje", col_min_trab]].copy()
    df_feat.columns = ["cantidad", "minutaje", "min_trab"]

    # evitar columnas duplicadas por seguridad
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

    df_feat = df_feat.apply(pd.to_numeric, errors="coerce")
    df_feat = df_feat[(df_feat["cantidad"] > 0) & (df_feat["minutaje"] > 0) & (df_feat["min_trab"] > 0)].copy()
    if df_feat.empty:
        return df_feat

    df_feat["minutos_producidos"] = df_feat["minutaje"] * df_feat["cantidad"]
    df_feat["eficiencia_pct"] = (df_feat["minutos_producidos"] / df_feat["min_trab"]) * 100
    df_feat["eficiencia"] = df_feat["eficiencia_pct"] / 100.0

    df_feat = df_feat[(df_feat["eficiencia_pct"] >= 0) & (df_feat["eficiencia_pct"] <= 120)].copy()

    bins = [0, 70, 85, 100]
    labels = ["Baja", "Media", "Alta"]
    df_feat["categoria"] = pd.cut(df_feat["eficiencia_pct"], bins=bins, labels=labels, include_lowest=True)

    return df_feat

def cargar_metricas_modelo(nombre_mostrado: str):
    cfg = MODELOS.get(nombre_mostrado)
    if not cfg:
        return None
    ruta = os.path.join(MODELS_DIR, cfg["metricas_file"])
    if os.path.exists(ruta):
        try:
            return joblib.load(ruta)
        except Exception:
            return None
    return None

def construir_tabla_metricas():
    filas = []
    for nombre_mostrado, cfg in MODELOS.items():
        m = cargar_metricas_modelo(nombre_mostrado)
        if m:
            filas.append([nombre_mostrado, m.get("accuracy", 0), m.get("precision", 0),
                          m.get("recall", 0), m.get("f1", 0), m.get("auc", 0)])
    if not filas:
        filas = [
            ["Random Forest (Bal)", 0.9787, 0.9842, 0.9787, 0.9811, 0.9906],
            ["Regresión Logística", 0.9592, 0.9823, 0.9592, 0.9683, 0.9874],
            ["SVM", 0.9823, 0.9901, 0.9823, 0.9848, 0.9934],
            ["Red Neuronal (ANN)", 0.9647, 0.9821, 0.9647, 0.9714, 0.9929],
        ]
    return pd.DataFrame(filas, columns=["Modelo", "Accuracy", "Precisión", "Recall", "F1-score", "AUC"])

def seleccionar_mejor_modelo(df_metricas: pd.DataFrame, criterio: str = "F1-score"):
    if df_metricas.empty or criterio not in df_metricas.columns:
        return None
    return df_metricas[criterio].idxmax()

def cargar_modelo_y_scaler(nombre_modelo_mostrado: str):
    cfg = MODELOS.get(nombre_modelo_mostrado)
    if not cfg:
        return None, None
    model_path = cfg["model_path"]
    scaler_path = cfg["scaler_path"]

    if not os.path.exists(model_path):
        return None, None
    try:
        modelo = joblib.load(model_path)
    except Exception:
        return None, None

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None
    return modelo, scaler

# ======================================
# TABLA DE MÉTRICAS Y SELECCIÓN AUTOMÁTICA
# ======================================
df_metricas_global = construir_tabla_metricas().set_index("Modelo")
mejor_modelo = seleccionar_mejor_modelo(df_metricas_global, "F1-score") or "Random Forest (Bal)"

# ======================================
# 1) RESUMEN GENERAL
# ======================================
if opcion == "Resumen general":
    st.subheader("Métricas de Evaluación de Modelos de Clasificación")
    st.dataframe(df_metricas_global.style.format("{:.4f}"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='metric-box'><h3>Modelo Seleccionado</h3><p><b>{mejor_modelo}</b></p></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='metric-box'><h3>Mayor Exactitud Global</h3><p><b>Accuracy ≈ {df_metricas_global['Accuracy'].max():.3f}</b></p></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        "<div class='metric-box'><h3>Respaldos</h3><p>Otros modelos sirven como referencias técnicas.</p></div>",
        unsafe_allow_html=True,
    )

# ======================================
# 2) SISTEMA PREDICTIVO (automático con mejor modelo)
# ======================================
elif opcion == "Sistema predictivo":
    st.subheader("Aplicación del modelo seleccionado sobre datos de producción")
    st.markdown(
        f"El sistema usa automáticamente **{mejor_modelo}** (mejor F1-score). "
        "Carga un Excel (.xlsx) o usa el dataset interno."
    )

    archivo = st.file_uploader("Cargar archivo (.xlsx) con datos de producción", type=["xlsx"])

    df_raw = None
    if archivo is not None:
        try:
            xls = pd.ExcelFile(archivo)
            df_raw = pd.concat([xls.parse(h) for h in xls.sheet_names], ignore_index=True)
        except Exception:
            st.info("No se pudo leer el archivo cargado. Verifique el formato.")
    else:
        if os.path.exists(DATA_PATH_DEFAULT):
            try:
                df_raw = cargar_datos_default()
            except Exception:
                st.info("No se pudo cargar el dataset interno.")
        else:
            st.info("No se encontró el dataset interno. Cargue un archivo manualmente.")

    if df_raw is not None:
        df_proc = preprocesar(df_raw)
        if df_proc.empty:
            st.info("No se encontraron registros válidos tras aplicar la fórmula oficial.")
        else:
            modelo_pred, scaler_pred = cargar_modelo_y_scaler(mejor_modelo)
            if modelo_pred is None:
                st.info(
                    f"No se encontró un modelo válido para '{mejor_modelo}' en '{MODELS_DIR}'. "
                    "Verifique las rutas y archivos .joblib."
                )
            else:
                X = df_proc[["cantidad", "minutaje", "min_trab"]].copy()
                try:
                    X_scaled = scaler_pred.transform(X) if scaler_pred is not None else X.values
                    pred_labels = modelo_pred.predict(X_scaled)
                except Exception as e:
                    st.info(f"No fue posible generar predicciones con {mejor_modelo}. Detalle técnico: {e}")
                    pred_labels = None

                if pred_labels is not None:
                    df_res = df_proc.copy()
                    df_res["pred_categoria"] = pred_labels

                    st.markdown("Vista preliminar de registros clasificados:")
                    st.dataframe(
                        df_res[["cantidad", "minutaje", "min_trab", "eficiencia_pct", "pred_categoria"]].head(30),
                        use_container_width=True,
                    )

                    conteo = df_res["pred_categoria"].value_counts().reindex(["Baja", "Media", "Alta"]).fillna(0).astype(int)
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='metric-box'><h3>Registros en Baja</h3><p><b>{conteo.get('Baja',0)}</b></p></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-box'><h3>Registros en Media</h3><p><b>{conteo.get('Media',0)}</b></p></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='metric-box'><h3>Registros en Alta</h3><p><b>{conteo.get('Alta',0)}</b></p></div>", unsafe_allow_html=True)

                    st.markdown("Distribución de niveles de eficiencia previstos:")
                    dist_df = pd.DataFrame(
                        {"Nivel": ["Baja", "Media", "Alta"],
                         "Cantidad": [conteo.get("Baja",0), conteo.get("Media",0), conteo.get("Alta",0)]}
                    ).set_index("Nivel")
                    st.bar_chart(dist_df)

                    # -------- Curva (FIX sin “línea plana”) ----------
                    st.markdown("Curva de aprendizaje estimada (eficiencia promedio móvil):")
                    curva = df_res.sort_index().copy()  # orden original
                    ventana = max(5, len(curva)//15)
                    curva["eficiencia_media_movil"] = (
                        curva["eficiencia_pct"].rolling(window=ventana, min_periods=3, center=True).mean()
                    )
                    curva_plot = curva[["eficiencia_media_movil"]].dropna().reset_index(drop=True)
                    curva_plot.index.name = "registro"
                    if not curva_plot.empty:
                        st.line_chart(curva_plot)
                        st.caption("Curva suavizada en el orden original de llegada (evita mesetas por x duplicadas).")
                    else:
                        st.caption("No se pudo calcular una curva suavizada por escasez de datos válidos.")

# ======================================
# 3) COMPARACIÓN DE MODELOS
# ======================================
elif opcion == "Comparación de modelos":
    st.subheader("Comparativa de modelos de clasificación")

    st.dataframe(df_metricas_global.style.format("{:.4f}"), use_container_width=True)

    # Figuras comparativas (si existen)
    comp_dir = os.path.join(FIG_DIR, "modelos_clasificacion")
    alt1 = os.path.join(FIG_DIR, "comparativas_class", "ROC_Comparativa_Modelos.png")

    mostradas = 0
    # heatmap y barras de F1 si están en /figuras/modelos_clasificacion
    for fname, cap in [
        ("heatmap_metricas_modelos.png", "Heatmap de métricas por modelo"),
        ("f1score_comparativa_final.png", "Comparativa de F1-score por modelo"),
    ]:
        p = os.path.join(comp_dir, fname)
        if _show_image_if_exists(p, cap):
            mostradas += 1

    # Curvas ROC simuladas si están en /figuras/comparativas_class
    if _show_image_if_exists(alt1, "Curvas ROC simuladas a partir de AUC"):
        mostradas += 1

    if mostradas == 0:
        st.warning(
            "No se encontraron imágenes de comparación en el repositorio.\n\n"
            "Súbelas a:\n"
            "- `figuras/modelos_clasificacion/heatmap_metricas_modelos.png`\n"
            "- `figuras/modelos_clasificacion/f1score_comparativa_final.png`\n"
            "- `figuras/comparativas_class/ROC_Comparativa_Modelos.png`"
        )

    if mejor_modelo:
        st.info(
            f"Según las métricas cargadas, el modelo con mejor F1-score es **{mejor_modelo}**. "
            "Los demás modelos sirven como respaldo y comparación."
        )
    else:
        st.info(
            "Aún no se encontraron métricas reales en la carpeta de modelos; "
            "se muestran valores de ejemplo para ilustrar la comparación."
        )

# ======================================
# 4) CURVAS DE ENTRENAMIENTO
# ======================================
elif opcion == "Curvas de entrenamiento por modelo":
    st.subheader("Curvas de entrenamiento/validación por modelo")

    curvas_dir = os.path.join(FIG_DIR, "curvas_modelos")
    candidatos = [
        (os.path.join(curvas_dir, "curvas_4_modelos.png"), "Curvas globales (4 modelos)"),
        (os.path.join(curvas_dir, "rf_curvas.png"), "Random Forest"),
        (os.path.join(curvas_dir, "svm_curvas.png"), "SVM"),
        (os.path.join(curvas_dir, "log_curvas.png"), "Regresión Logística"),
        (os.path.join(curvas_dir, "ann_curvas.png"), "Red Neuronal (ANN)"),
    ]

    mostradas = 0
    for p, cap in candidatos:
        if _show_image_if_exists(p, cap):
            mostradas += 1

    if mostradas == 0:
        st.warning(
            "No se encontraron imágenes de curvas en el repositorio.\n\n"
            "Súbelas a `figuras/curvas_modelos/` con estos nombres opcionales:\n"
            "- `curvas_4_modelos.png`, `rf_curvas.png`, `svm_curvas.png`,\n"
            "  `log_curvas.png`, `ann_curvas.png`.\n\n"
            "Si prefieres generarlas on-the-fly, avísame y lo integramos aquí."
        )


# ======================================
# 4) INFORMACIÓN DEL PROYECTO
# ======================================
elif opcion == "Información del proyecto":
    st.subheader("Información del Proyecto")
    st.markdown("""
**Proyecto:** Modelo predictivo aplicando Machine Learning para la identificación de la curva de aprendizaje en la producción textil.  
**Cliente:** Topitop S.A.  
**Equipo de desarrollo:** Hector Agustín Garcia Cortez - Jorge Hiro Chung Quispe  
**Institución:** Universidad Privada del Norte – Ingeniería de Sistemas Computacionales – 2025  
**Metodología:** CRISP–DM  
**Entorno:** Python 3.10 – Visual Studio Code  
**Bibliotecas:** pandas, scikit-learn, TensorFlow/Keras, seaborn, matplotlib, streamlit  
""")
    st.info(
        "Este panel aplica la fórmula oficial (minutaje*cantidad / min_trab * 100), "
        "selecciona automáticamente el mejor modelo por F1 y muestra la curva de aprendizaje en tiempo real."
    )
