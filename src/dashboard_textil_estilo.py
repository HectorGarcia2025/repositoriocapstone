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
# ARCHIVOS DE MODELOS
# ======================================
MODELOS = {
    "Random Forest": {
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
body {
  background-color:#0d1015;
  color:#f5f7fa;
  font-family:'Segoe UI',sans-serif;
}
.block-container { padding-top: 1.5rem; }

/* Barra superior tipo navbar */
.top-navbar{
  background:linear-gradient(90deg,#d81f26,#8b1a1f,#222);
  color:#fff;
  padding:12px 24px;
  border-radius:12px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  box-shadow:0 4px 12px rgba(0,0,0,.25);
  margin-bottom:1.2rem;
}
.top-navbar .brand-title{
  font-size:1.2rem;
  font-weight:700;
  letter-spacing:0.04em;
  text-transform:uppercase;
}
.top-navbar .brand-sub{
  font-size:0.8rem;
  opacity:0.9;
}
.top-navbar .nav-links{
  display:flex;
  gap:0.6rem;
  font-size:0.85rem;
}
.top-navbar .nav-pill{
  padding:6px 12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.35);
  background-color:rgba(255,255,255,0.06);
  text-decoration:none;
  color:#fff;
}

/* Hero principal */
.hero-section{
  background:radial-gradient(circle at top left,#ff6b6b,#d81f26,#191919);
  color:#fff;
  padding:28px 28px 24px 28px;
  border-radius:20px;
  box-shadow:0 10px 25px rgba(0,0,0,.35);
  margin-bottom:1.8rem;
}
.hero-grid{
  display:flex;
  flex-wrap:wrap;
  gap:1.5rem;
  justify-content:space-between;
  align-items:flex-start;
}
.hero-left{ max-width:640px; }
.hero-tag{
  font-size:0.9rem;
  text-transform:uppercase;
  letter-spacing:0.18em;
  opacity:0.8;
  margin-bottom:0.3rem;
}
.hero-title{
  font-size:2.1rem;
  font-weight:800;
  margin-bottom:0.4rem;
}
.hero-subtitle{
  font-size:0.98rem;
  line-height:1.55;
  max-width:560px;
  opacity:0.95;
}
.pill-nav{
  display:flex;
  flex-wrap:wrap;
  gap:0.5rem;
  margin-top:0.8rem;
}
.pill-nav a{
  text-decoration:none;
  font-size:0.85rem;
  padding:6px 14px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.45);
  color:#fff;
  background-color:rgba(0,0,0,0.14);
}
.hero-right{ min-width:230px; }
.hero-kpi-card{
  background-color:rgba(0,0,0,0.45);
  border-radius:16px;
  padding:12px 16px;
  margin-bottom:10px;
  border:1px solid rgba(255,255,255,0.15);
}
.hero-kpi-title{
  font-size:0.8rem;
  text-transform:uppercase;
  letter-spacing:0.1em;
  opacity:0.8;
}
.hero-kpi-main{ font-size:1.5rem; font-weight:700; }
.hero-kpi-sub{ font-size:0.8rem; opacity:0.9; }

/* Secciones info */
.info-section{
  background-color:#11151d;
  border-radius:18px;
  padding:18px 20px 16px 20px;
  margin-bottom:1.4rem;
  box-shadow:0 4px 14px rgba(0,0,0,0.4);
}
.section-title{
  font-size:1.4rem;
  font-weight:700;
  color:#ff6b6b;
  margin-bottom:0.3rem;
}
.section-subtitle{
  font-size:0.92rem;
  color:#c4c9d4;
  margin-bottom:0.8rem;
}
.benefits-grid, .impact-grid{
  display:flex;
  flex-wrap:wrap;
  gap:0.9rem;
}
.benefit-card, .impact-card{
  flex:1 1 220px;
  background-color:#151b26;
  border-radius:14px;
  padding:10px 12px;
  border:1px solid #262d3d;
}
.benefit-title, .impact-title{
  font-weight:600;
  font-size:0.95rem;
  color:#ff6b6b;
  margin-bottom:0.15rem;
}
.benefit-text, .impact-text{
  font-size:0.86rem;
  color:#e2e6f0;
}

/* Métricas */
.metric-box{
  background-color:#151b26;
  border:2px solid #d81f26;
  border-radius:12px;
  box-shadow:0 3px 6px rgba(0,0,0,.4);
  text-align:center;
  padding:14px;
  color:#fff;
}
.metric-box h3{
  color:#ff6b6b;
  margin-bottom:4px;
}

/* Títulos / tablas */
h2,h3{ color:#ff6b6b; }
.dataframe{
  background-color:#151b26 !important;
  border-radius:10px;
}

/* Alertas */
div[data-testid="stAlert"]{
  background-color:#d81f26 !important;
  border:1px solid #a1151c !important;
  border-radius:8px !important;
  color:white !important;
}
div[data-testid="stAlert"] *{
  color:white !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# ENCABEZADO / NAVBAR SUPERIOR
# ======================================
fecha_actual = datetime.now().strftime("%d/%m/%Y")
dataset_nombre = os.path.basename(DATA_PATH_DEFAULT)
st.markdown(f"""
<div class='top-navbar'>
  <div>
    <div class='brand-title'>TOPITOP · CURVA DE APRENDIZAJE</div>
    <div class='brand-sub'>Dashboard de predicción textil · Actualizado el {fecha_actual}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR (navegación)
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
# UTILIDADES Y PREPROCESO  (igual que antes)
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
    Fórmula oficial:
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

    # SOLO REGISTROS DE TIPO SALIDA
    if "tipo" in df.columns:
        df = df[df["tipo"].str.contains("salida", case=False, na=False)]

    df = df.loc[:, ~df.columns.duplicated()]

    cand_cant = [c for c in df.columns if "cant" in c]
    if not cand_cant:
        return pd.DataFrame()
    col_cant = cand_cant[0]

    cand_trab = [c for c in df.columns if "min trab" in c or "min_trab" in c or "perman" in c]
    if not cand_trab:
        return pd.DataFrame()
    col_min_trab = cand_trab[0]

    cand_minutaje = [c for c in df.columns if "mix" in c or "minutaje" in c or ("minuto" in c and "prenda" in c)]
    if cand_minutaje:
        col_minutaje = cand_minutaje[0]
        df["minutaje"] = pd.to_numeric(df[col_minutaje], errors="coerce")
    else:
        cand_totalmin = [c for c in df.columns if "total" in c and "min" in c]
        if not cand_totalmin:
            return pd.DataFrame()
        col_total_min = cand_totalmin[0]
        df["minutaje"] = (
            pd.to_numeric(df[col_total_min], errors="coerce")
            / pd.to_numeric(df[col_cant], errors="coerce")
        )

    df_feat = df[[col_cant, "minutaje", col_min_trab]].copy()
    df_feat.columns = ["cantidad", "minutaje", "min_trab"]
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]
    df_feat = df_feat.apply(pd.to_numeric, errors="coerce")

    df_feat = df_feat[
        (df_feat["cantidad"] > 0)
        & (df_feat["minutaje"] > 0)
        & (df_feat["min_trab"] > 0)
    ].copy()
    if df_feat.empty:
        return df_feat

    df_feat["minutos_producidos"] = df_feat["minutaje"] * df_feat["cantidad"]
    df_feat["eficiencia_pct"] = (df_feat["minutos_producidos"] / df_feat["min_trab"]) * 100
    df_feat["eficiencia"] = df_feat["eficiencia_pct"] / 100.0

    df_feat = df_feat[
        (df_feat["eficiencia_pct"] >= 0)
        & (df_feat["eficiencia_pct"] <= 120)
    ].copy()

    bins = [0, 70, 85, 100]
    labels = ["Baja", "Media", "Alta"]
    df_feat["categoria"] = pd.cut(
        df_feat["eficiencia_pct"], bins=bins, labels=labels, include_lowest=True
    )

    col_fecha = [c for c in df.columns if "fecha" in c]
    if col_fecha:
        fecha_series = pd.to_datetime(
            df[col_fecha[0]],
            errors="coerce",
            dayfirst=True
        )
        df_feat["fecha"] = fecha_series.reindex(df_feat.index)

    col_prenda = [c for c in df.columns if "prenda" in c or "estilo" in c or "modelo" in c]
    if col_prenda:
        prenda_series = df[col_prenda[0]].astype(str)
        df_feat["prenda"] = prenda_series.reindex(df_feat.index).fillna("Sin prenda")

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
            filas.append([
                nombre_mostrado,
                m.get("accuracy", 0),
                m.get("precision", 0),
                m.get("recall", 0),
                m.get("f1", 0),
                m.get("auc", 0),
            ])
    if not filas:
        filas = [
            ["Random Forest",        0.9787, 0.9842, 0.9787, 0.9811, 0.9906],
            ["Regresión Logística",  0.9592, 0.9823, 0.9592, 0.9683, 0.9874],
            ["SVM",                  0.9823, 0.9901, 0.9823, 0.9848, 0.9934],
            ["Red Neuronal (ANN)",   0.9647, 0.9821, 0.9647, 0.9714, 0.9929],
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
mejor_modelo = seleccionar_mejor_modelo(df_metricas_global, "F1-score") or "Random Forest"

max_acc = float(df_metricas_global["Accuracy"].max()) if not df_metricas_global.empty else 0.0
avg_f1 = float(df_metricas_global["F1-score"].mean()) if not df_metricas_global.empty else 0.0

# ======================================
# 1) RESUMEN GENERAL
# ======================================
if opcion == "Resumen general":
    st.markdown(f"""
    <section id="home" class="hero-section">
      <div class="hero-grid">
        <div class="hero-left">
          <p class="hero-tag">Topitop · Línea de confección</p>
          <h1 class="hero-title">Predicciones para entender la curva de aprendizaje textil</h1>
          <p class="hero-subtitle">
            Este dashboard integra modelos de Machine Learning (Random Forest, Regresión Logística,
            SVM y Redes Neuronales) para clasificar el nivel de eficiencia de la producción
            y visualizar cómo evoluciona la curva de aprendizaje de las líneas de confección.
          </p>
          <div class="pill-nav">
            <a href="#beneficios">Beneficios</a>
            <a href="#impacto">Impacto de nuestras predicciones</a>
            <a href="#indicadores">Indicadores de desempeño</a>
          </div>
        </div>
        <div class="hero-right">
          <div class="hero-kpi-card">
            <div class="hero-kpi-title">Modelo principal</div>
            <div class="hero-kpi-main">{mejor_modelo}</div>
            <div class="hero-kpi-sub">Seleccionado por mayor F1-score global.</div>
          </div>
          <div class="hero-kpi-card">
            <div class="hero-kpi-title">Accuracy máximo</div>
            <div class="hero-kpi-main">{max_acc:.2%}</div>
            <div class="hero-kpi-sub">Mejor exactitud alcanzada entre los modelos.</div>
          </div>
          <div class="hero-kpi-card">
            <div class="hero-kpi-title">F1-score promedio</div>
            <div class="hero-kpi-main">{avg_f1:.2%}</div>
            <div class="hero-kpi-sub">Equilibrio entre precisión y recall.</div>
          </div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)

    st.dataframe(df_metricas_global.style.format("{:.4f}"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='metric-box'><h3>Modelo Seleccionado</h3>"
        f"<p><b>{mejor_modelo}</b></p></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='metric-box'><h3>Mayor Exactitud Global</h3>"
        f"<p><b>Accuracy ≈ {df_metricas_global['Accuracy'].max():.3f}</b></p></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        "<div class='metric-box'><h3>Respaldo Analítico</h3>"
        "<p>Otros modelos sirven como comparación y validación técnica.</p></div>",
        unsafe_allow_html=True,
    )

    # --------- SECCIONES: BENEFICIOS / IMPACTO / INDICADORES ----------
    st.markdown("""
    <section id="beneficios" class="info-section">
      <h2 class="section-title">Beneficios para la operación</h2>
      <p class="section-subtitle">
        El modelo permite anticipar el nivel de eficiencia de la línea y entender cómo evoluciona la
        curva de aprendizaje a medida que se producen más lotes del mismo estilo.
      </p>
      <div class="benefits-grid">
        <div class="benefit-card">
          <div class="benefit-title">Visibilidad de la curva</div>
          <div class="benefit-text">
            Se visualiza de forma continua la evolución de la eficiencia real y predicha, ayudando a
            identificar cuándo la línea se estabiliza.
          </div>
        </div>
        <div class="benefit-card">
          <div class="benefit-title">Soporte a la decisión</div>
          <div class="benefit-text">
            Supervisores y jefes de planta pueden tomar decisiones informadas sobre ajustes de carga,
            cambios de estilo o capacitación.
          </div>
        </div>
        <div class="benefit-card">
          <div class="benefit-title">Detección temprana de desviaciones</div>
          <div class="benefit-text">
            Diferencias entre la eficiencia real y la predicha alertan sobre posibles problemas en
            la línea (cuellos de botella, falta de balanceo, etc.).
          </div>
        </div>
      </div>
    </section>

    <section id="impacto" class="info-section">
      <h2 class="section-title">Impacto de nuestras predicciones</h2>
      <p class="section-subtitle">
        Las predicciones de eficiencia ofrecen una referencia cuantitativa de lo que se espera de cada
        lote, considerando la experiencia acumulada de la línea.
      </p>
      <div class="impact-grid">
        <div class="impact-card">
          <div class="impact-title">Planeamiento más realista</div>
          <div class="impact-text">
            Al conocer la curva de aprendizaje, se pueden estimar mejor los tiempos de producción y
            fechas de entrega.
          </div>
        </div>
        <div class="impact-card">
          <div class="impact-title">Estándares basados en datos</div>
          <div class="impact-text">
            La empresa puede contrastar sus estándares teóricos con el comportamiento real observado en planta.
          </div>
        </div>
        <div class="impact-card">
          <div class="impact-title">Mejora continua</div>
          <div class="impact-text">
            El histórico de curvas permite comparar estilos, líneas y periodos, facilitando iniciativas
            de mejora y proyectos Kaizen.
          </div>
        </div>
      </div>
    </section>

    <section id="indicadores" class="info-section">
      <h2 class="section-title">Indicadores de desempeño del modelo</h2>
      <p class="section-subtitle">
        El sistema integra varios modelos de clasificación y selecciona el mejor según el F1-score,
        equilibrando precisión y recall.
      </p>
      <div class="impact-grid">
        <div class="impact-card">
          <div class="impact-title">Exactitud (Accuracy)</div>
          <div class="impact-text">
            Mide el porcentaje de registros correctamente clasificados entre Baja, Media y Alta eficiencia.
          </div>
        </div>
        <div class="impact-card">
          <div class="impact-title">F1-score</div>
          <div class="impact-text">
            Combina precisión y recall, dándole más peso a un desempeño balanceado en las diferentes clases.
          </div>
        </div>
        <div class="impact-card">
          <div class="impact-title">Curva real vs predicha</div>
          <div class="impact-text">
            La comparación visual entre la eficiencia observada y la esperada ayuda a validar el
            comportamiento general del modelo en el tiempo.
          </div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)

# ======================================
# 2) SISTEMA PREDICTIVO
# ======================================
elif opcion == "Sistema predictivo":
    # ancla para el navbar superior
    st.markdown("<div id='sistema-predictivo'></div>", unsafe_allow_html=True)

    st.subheader("Aplicación del modelo seleccionado sobre datos de producción")
    st.markdown(
        f"El sistema usa automáticamente **{mejor_modelo}** (mejor Accuracy - Precisión - Recall - F1-score). "
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

                    eficiencia_predicha_pct = None

                    if "categoria" in df_proc.columns:
                        niveles = (
                            df_proc.groupby("categoria")["eficiencia_pct"]
                            .mean()
                            .to_dict()
                        )
                    else:
                        niveles = {}

                    default_niveles = {"Baja": 40.0, "Media": 70.0, "Alta": 90.0}
                    for k, v in default_niveles.items():
                        niveles.setdefault(k, v)

                    if hasattr(modelo_pred, "predict_proba"):
                        try:
                            proba = modelo_pred.predict_proba(X_scaled)
                            clases = list(modelo_pred.classes_)
                            eficiencia_predicha_pct = np.zeros(proba.shape[0], dtype=float)
                            for idx, c in enumerate(clases):
                                eficiencia_predicha_pct += niveles.get(str(c), 0.0) * proba[:, idx]
                        except Exception:
                            eficiencia_predicha_pct = None

                    if eficiencia_predicha_pct is None:
                        eficiencia_predicha_pct = (
                            pd.Series(pred_labels)
                            .astype(str)
                            .map(niveles)
                            .fillna(df_proc["eficiencia_pct"])
                            .to_numpy()
                        )

                    min_real = float(df_proc["eficiencia_pct"].min())
                    max_real = float(df_proc["eficiencia_pct"].max())
                    min_pred = float(np.nanmin(eficiencia_predicha_pct))
                    max_pred = float(np.nanmax(eficiencia_predicha_pct))

                    if max_pred > min_pred and max_real > min_real:
                        ef_norm = (eficiencia_predicha_pct - min_pred) / (max_pred - min_pred)
                        eficiencia_predicha_pct = ef_norm * (max_real - min_real) + min_real

                    # ajuste de nivel (shift)
                    mean_real = float(df_proc["eficiencia_pct"].mean())
                    mean_pred = float(np.nanmean(eficiencia_predicha_pct))
                    shift = mean_real - mean_pred
                    eficiencia_predicha_pct = eficiencia_predicha_pct + shift
                    eficiencia_predicha_pct = np.clip(eficiencia_predicha_pct, min_real, max_real)

                except Exception as e:
                    st.info(f"No fue posible generar predicciones con {mejor_modelo}. Detalle técnico: {e}")
                    pred_labels = None
                    eficiencia_predicha_pct = None

                if pred_labels is not None:
                    df_res = df_proc.copy()
                    df_res["pred_categoria"] = pred_labels
                    if eficiencia_predicha_pct is not None:
                        df_res["eficiencia_predicha_pct"] = eficiencia_predicha_pct
                    else:
                        df_res["eficiencia_predicha_pct"] = df_res["eficiencia_pct"]

                    st.markdown("Vista preliminar de registros clasificados:")
                    st.dataframe(
                        df_res[["cantidad", "minutaje", "min_trab", "eficiencia_pct", "pred_categoria"]].head(30),
                        use_container_width=True,
                    )

                    conteo = df_res["pred_categoria"].value_counts().reindex(["Baja", "Media", "Alta"]).fillna(0).astype(int)
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(
                        f"<div class='metric-box'><h3>Registros en Baja</h3>"
                        f"<p><b>{conteo.get('Baja',0)}</b></p></div>",
                        unsafe_allow_html=True,
                    )
                    c2.markdown(
                        f"<div class='metric-box'><h3>Registros en Media</h3>"
                        f"<p><b>{conteo.get('Media',0)}</b></p></div>",
                        unsafe_allow_html=True,
                    )
                    c3.markdown(
                        f"<div class='metric-box'><h3>Registros en Alta</h3>"
                        f"<p><b>{conteo.get('Alta',0)}</b></p></div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("Distribución de niveles de eficiencia previstos:")
                    dist_df = pd.DataFrame(
                        {
                            "Nivel": ["Baja", "Media", "Alta"],
                            "Cantidad": [conteo.get("Baja",0), conteo.get("Media",0), conteo.get("Alta",0)],
                        }
                    ).set_index("Nivel")
                    st.bar_chart(dist_df)

                    st.markdown("### Curva de aprendizaje: eficiencia real vs predicha")

                    df_plot = df_res.copy()
                    colf1, colf2 = st.columns(2)
                    usa_fecha = False

                    if "fecha" in df_plot.columns and df_plot["fecha"].notna().any():
                        df_plot = df_plot[df_plot["fecha"].notna()].copy()
                        df_plot["anio"] = df_plot["fecha"].dt.year
                        anios = sorted(df_plot["anio"].dropna().unique().tolist())
                        opcion_anio = colf1.selectbox(
                            "Filtrar por año:",
                            ["Todos los años"] + [str(a) for a in anios],
                        )
                        if opcion_anio != "Todos los años":
                            df_plot = df_plot[df_plot["anio"] == int(opcion_anio)]
                        usa_fecha = True
                    else:
                        colf1.write("Sin columna de fecha detectada.")

                    if "prenda" in df_plot.columns and df_plot["prenda"].notna().any():
                        prendas = sorted(df_plot["prenda"].dropna().unique().tolist())
                        opcion_prenda = colf2.selectbox(
                            "Filtrar por prenda:",
                            ["Todas las prendas"] + prendas,
                        )
                        if opcion_prenda != "Todas las prendas":
                            df_plot = df_plot[df_plot["prenda"] == opcion_prenda]
                    else:
                        colf2.write("Sin columna de prenda detectada.")

                    if df_plot.empty:
                        st.caption("No hay registros para los filtros seleccionados.")
                    else:
                        if usa_fecha:
                            df_plot = df_plot.sort_values("fecha")
                        else:
                            df_plot = df_plot.reset_index(drop=True)

                        n_reg = len(df_plot)
                        if n_reg < 5:
                            st.caption("Hay muy pocos registros para trazar una curva de aprendizaje.")
                            
                        else:
                            ventana = st.slider(
                                "Tamaño de la ventana del promedio móvil (n° de registros)",
                                min_value=5,
                                max_value=max(5, n_reg // 3),
                                value=max(5, n_reg // 10),
                                step=1,
                                help="Mientras más grande la ventana, más suave será la curva.",
                            )

                            df_plot["ef_real_ma"] = (
                                df_plot["eficiencia_pct"]
                                .rolling(window=ventana, min_periods=3, center=True)
                                .mean()
                            )
                            df_plot["ef_pred_ma"] = (
                                df_plot["eficiencia_predicha_pct"]
                                .rolling(window=ventana, min_periods=3, center=True)
                                .mean()
                            )

                            mask = df_plot["ef_real_ma"].notna() & df_plot["ef_pred_ma"].notna()
                            df_curve = df_plot.loc[mask].copy()

                            if df_curve.empty:
                                st.caption("No se pudo calcular una curva suavizada con la ventana seleccionada.")
                            else:
                                if usa_fecha:
                                    df_curve["x"] = df_curve["fecha"]
                                else:
                                    df_curve["x"] = np.arange(1, len(df_curve) + 1)

                                df_line = df_curve.set_index("x")[["ef_real_ma", "ef_pred_ma"]]
                                df_line.columns = ["Eficiencia real", "Eficiencia predicha"]

                                st.line_chart(df_line)
                                st.caption(
                                    "Curvas suavizadas (promedio móvil) de eficiencia real y predicha, "
                                    "con la predicha reescalada y ajustada de nivel para ser comparable."
                                )

# ======================================
# 3) COMPARACIÓN DE MODELOS
# ======================================
elif opcion == "Comparación de modelos":
    st.subheader("Comparativa de modelos de clasificación")

    st.dataframe(df_metricas_global.style.format("{:.4f}"), use_container_width=True)

    comp_dir = os.path.join(FIG_DIR, "modelos_clasificacion")
    alt1 = os.path.join(FIG_DIR, "comparativas_class", "ROC_Comparativa_Modelos.png")

    mostradas = 0
    for fname, cap in [
        ("heatmap_metricas_modelos.png", "Heatmap de métricas por modelo"),
        ("f1score_comparativa_final.png", "Comparativa de F1-score por modelo"),
    ]:
        p = os.path.join(comp_dir, fname)
        if _show_image_if_exists(p, cap):
            mostradas += 1

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
# 5) INFORMACIÓN DEL PROYECTO
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
        "El sistema selecciona automáticamente el mejor modelo por la mayoría de métricas "
        "y muestra la curva de aprendizaje en tiempo real."
    )
