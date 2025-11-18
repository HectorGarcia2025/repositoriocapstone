import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib

# ======================================
# CONFIGURACIÓN GENERAL
# ======================================
st.set_page_config(
    page_title="Dashboard Curva de Aprendizaje - Topitop",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH_DEFAULT = os.path.join(BASE_DIR, "data", "2 Salida de prendas.xlsx")

# Rutas de modelos (ajusta si cambian)
MODEL_RF_PATH   = os.path.join(BASE_DIR, "models", "modelo_curva_rf_class_bal.joblib")
SCALER_RF_PATH  = os.path.join(BASE_DIR, "models", "scaler_X_rf_class_bal.joblib")

# ======================================
# ESTILOS CSS
# ======================================
st.markdown("""
<style>
body { background-color:#f7f9fb; color:#2e3b4e; font-family:'Segoe UI',sans-serif; }
.main-header {
  background: linear-gradient(90deg,#d81f26,#222); color:white; padding:15px 25px;
  border-radius:8px; display:flex; justify-content:space-between; align-items:center;
}
.main-header img { height:55px; }
.metric-box {
  background:#fff; border:2px solid #d81f26; border-radius:10px; box-shadow:0 3px 6px rgba(0,0,0,.1);
  text-align:center; padding:15px; color:#000;
}
.metric-box h3 { color:#d81f26; margin-bottom:6px; }
h2, h3 { color:#d81f26; }
.dataframe { background:#fff !important; border-radius:10px; }
/* Forzar cuadros info a rojo con texto blanco */
div[data-testid="stAlert"] { background:#d81f26 !important; border:1px solid #a1151c !important; border-radius:8px !important; color:white !important; }
div[data-testid="stAlert"] * { color:white !important; }
</style>
""", unsafe_allow_html=True)

# ======================================
# ENCABEZADO
# ======================================
fecha_actual = datetime.now().strftime("%d/%m/%Y")
st.markdown(f"""
<div class='main-header'>
  <div>
    <h1>Modelo Predictivo de Curva de Aprendizaje Textil</h1>
    <p>Sistema prototipo de apoyo a la decisión - Empresa Topitop</p>
    <small>Actualización: {fecha_actual} &nbsp;|&nbsp; Dataset base: 2 Salida de prendas.xlsx</small>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================================
# SIDEBAR
# ======================================
st.sidebar.image("src/topitop_logo.png", width=150)
st.sidebar.header("Navegación")

opcion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Resumen general",
        "Sistema predictivo (RF)",
        "Comparación de modelos",
        "Mapas de calor",
        "Análisis avanzado",
        "Información del proyecto"
    ]
)

# ======================================
# FUNCIONES AUXILIARES
# ======================================
def cargar_datos_default():
    hojas = ["L72", "L79"]
    df_list = []
    for h in hojas:
        df_hoja = pd.read_excel(DATA_PATH_DEFAULT, sheet_name=h)
        df_list.append(df_hoja)
    return pd.concat(df_list, ignore_index=True)

def _find_col(df, pred):
    try:
        return [c for c in df.columns if pred(c)][0]
    except IndexError:
        return None

def preprocesar(df_raw: pd.DataFrame, tol_discrep=0.05):
    """
    Plantilla alineada con tu archivo:
    - H: Minutaje (minuto por prenda)   -> 'minuto prenda'
    - J: Cantidad                       -> 'cantidad'
    - M: Total Min (minutos producidos) -> 'minutos producidos' (si no existe, se calcula)
    - N: Min Trab (minutos permanencia) -> 'minutos permanencia'
    Eficiencia% = (minutos producidos / minutos permanencia) * 100
    Filtro: valores <=0 o NaN se eliminan. Outliers >120% se quitan.
    Además: valida discrepancias entre Total Min y Minutaje×Cantidad.
    """
    meta = {"n_total": 0, "n_validos": 0, "n_con_totalmin": 0,
            "n_con_minutaje": 0, "n_discrepancias": 0, "mapd_discrepancia": None}

    df = df_raw.copy()
    df.columns = (df.columns.str.strip()
                            .str.lower()
                            .str.replace("'", "")
                            .str.replace('"', ""))

    if "tipo" in df.columns:
        df = df[df["tipo"].str.contains("salida", case=False, na=False)]

    meta["n_total"] = len(df)

    col_cant  = _find_col(df, lambda c: "cant" in c)  # J
    col_minut = _find_col(df, lambda c: ("minut" in c) and ("perman" not in c) and ("prod" not in c))  # H
    col_tot   = _find_col(df, lambda c: ("total" in c and "min" in c) or ("min" in c and "prod" in c))  # M
    col_perm  = _find_col(df, lambda c: "perman" in c or "trab" in c)  # N

    if not col_cant or not col_perm or (not col_tot and not col_minut):
        return pd.DataFrame(), meta  # faltan columnas clave

    use_cols = [x for x in [col_cant, col_minut, col_tot, col_perm] if x]
    df = df[use_cols].apply(pd.to_numeric, errors="coerce")

    mapping = {col_cant: "cantidad", col_perm: "minutos permanencia"}
    if col_minut: mapping[col_minut] = "minuto prenda"
    if col_tot:   mapping[col_tot]   = "minutos producidos"
    df = df.rename(columns=mapping)

    if "minutos producidos" in df.columns:
        meta["n_con_totalmin"] = df["minutos producidos"].notna().sum()
    if "minuto prenda" in df.columns:
        meta["n_con_minutaje"] = df["minuto prenda"].notna().sum()

    # Si no hay Total Min, calcularlo
    if "minutos producidos" not in df.columns and "minuto prenda" in df.columns:
        df["minutos producidos"] = df["minuto prenda"] * df["cantidad"]

    # Si existen ambas, validar discrepancias
    if "minutos producidos" in df.columns and "minuto prenda" in df.columns:
        calc_prod = df["minuto prenda"] * df["cantidad"]
        with np.errstate(divide='ignore', invalid='ignore'):
            diff_ratio = np.abs(df["minutos producidos"] - calc_prod) / np.where(df["minutos producidos"]!=0, df["minutos producidos"], np.nan)
        discrep_mask = diff_ratio > tol_discrep
        meta["n_discrepancias"] = int(discrep_mask.fillna(False).sum())
        meta["mapd_discrepancia"] = float(np.nanmean(diff_ratio)) if np.isfinite(np.nanmean(diff_ratio)) else None
        # Reemplazar Total Min por el calculado cuando falte o sea cero
        df.loc[(df["minutos producidos"].isna()) | (df["minutos producidos"]<=0), "minutos producidos"] = calc_prod

    # Filtrar inválidos
    df = df[
        (df["cantidad"] > 0) &
        (df["minutos producidos"] > 0) &
        (df["minutos permanencia"] > 0)
    ].copy()

    if df.empty:
        meta["n_validos"] = 0
        return df, meta

    # Eficiencia
    df["eficiencia_pct"] = (df["minutos producidos"] / df["minutos permanencia"]) * 100
    df["eficiencia"] = df["eficiencia_pct"] / 100.0

    # Outliers
    df = df[(df["eficiencia_pct"] >= 0) & (df["eficiencia_pct"] <= 120)].copy()

    # Asegurar minuto prenda (si solo vino Total Min)
    if "minuto prenda" not in df.columns:
        df["minuto prenda"] = df["minutos producidos"] / df["cantidad"]

    # Categorías
    bins = [0, 70, 85, 100]
    labels = ["Baja", "Media", "Alta"]
    df["categoria"] = pd.cut(df["eficiencia_pct"], bins=bins, labels=labels, include_lowest=True)

    meta["n_validos"] = len(df)

    # Orden amigable
    cols_view = ["cantidad", "minuto prenda", "minutos producidos", "minutos permanencia", "eficiencia_pct", "categoria"]
    df = df[cols_view]

    return df, meta

def cargar_modelo_rf():
    if not (os.path.exists(MODEL_RF_PATH) and os.path.exists(SCALER_RF_PATH)):
        return None, None
    return joblib.load(MODEL_RF_PATH), joblib.load(SCALER_RF_PATH)

def cargar_metricas_modelo(nombre_archivo):
    ruta = os.path.join(BASE_DIR, "models", nombre_archivo)
    if os.path.exists(ruta):
        try:
            return joblib.load(ruta)
        except Exception:
            return None
    return None

def construir_tabla_metricas():
    filas = []
    rf = cargar_metricas_modelo("metricas_rf_class_bal.joblib")
    if rf: filas.append(["Random Forest (Bal)", rf.get("accuracy",0), rf.get("precision",0), rf.get("recall",0), rf.get("f1",0), rf.get("auc",0)])
    log = cargar_metricas_modelo("metricas_log_class.joblib")
    if log: filas.append(["Regresión Logística", log.get("accuracy",0), log.get("precision",0), log.get("recall",0), log.get("f1",0), log.get("auc",0)])
    svm = cargar_metricas_modelo("metricas_svm_class.joblib")
    if svm: filas.append(["SVM", svm.get("accuracy",0), svm.get("precision",0), svm.get("recall",0), svm.get("f1",0), svm.get("auc",0)])
    ann = cargar_metricas_modelo("metricas_ann_class.joblib")
    if ann: filas.append(["Red Neuronal (ANN)", ann.get("accuracy",0), ann.get("precision",0), ann.get("recall",0), ann.get("f1",0), ann.get("auc",0)])

    if not filas:
        filas = [
            ["Random Forest (Bal)", 0.9388, 0.9455, 0.9388, 0.9393, 0.2765],
            ["Regresión Logística", 0.5793, 0.5836, 0.5793, 0.5813, 0.3331],
            ["SVM", 0.7655, 0.7898, 0.7655, 0.7603, 0.2938],
            ["Red Neuronal (ANN)", 0.7284, 0.7393, 0.7284, 0.7327, 0.8956],
        ]
    return pd.DataFrame(filas, columns=["Modelo","Accuracy","Precisión","Recall","F1-score","AUC"])

# ======================================
# 1) RESUMEN GENERAL
# ======================================
if opcion == "Resumen general":
    st.subheader("Métricas de Evaluación de Modelos de Clasificación")
    df_metricas = construir_tabla_metricas().set_index("Modelo")
    st.dataframe(df_metricas.style.format("{:.4f}"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='metric-box'><h3>Modelo Seleccionado</h3><p><b>Random Forest (Clasificación balanceada)</b></p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-box'><h3>Mayor Exactitud Global</h3><p><b>Accuracy ≈ 0.94</b></p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-box'><h3>Respaldos</h3><p>Logística, SVM y ANN.</p></div>", unsafe_allow_html=True)

    st.info("RF balanceado ofrece el mejor equilibrio entre precisión, recall y F1 para clasificar niveles de eficiencia (Baja, Media, Alta).")

# ======================================
# 2) SISTEMA PREDICTIVO (RF)
# ======================================
elif opcion == "Sistema predictivo (RF)":
    st.subheader("Aplicación del modelo Random Forest sobre datos de producción")
    st.markdown("Cargue un .xlsx de Topitop o use el dataset interno. El sistema recalcula la eficiencia con la fórmula oficial y clasifica la curva de aprendizaje.")

    tol_ui = st.sidebar.slider("Tolerancia de discrepancia TotalMin vs (Minutaje×Cantidad)", 0.0, 0.2, 0.05, 0.01)

    archivo = st.file_uploader("Cargar archivo (.xlsx)", type=["xlsx"])

    df_raw = None
    if archivo is not None:
        try:
            xls = pd.ExcelFile(archivo)
            df_list = [xls.parse(h) for h in xls.sheet_names]
            df_raw = pd.concat(df_list, ignore_index=True)
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
        df_proc, meta = preprocesar(df_raw, tol_discrep=tol_ui)

        # Resumen de calidad
        with st.expander("Resumen de consistencia del archivo (según fórmula de negocio)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-box'><h3>Registros brutos</h3><p><b>{meta['n_total']}</b></p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><h3>Válidos tras reglas</h3><p><b>{meta['n_validos']}</b></p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><h3>Con Total Min</h3><p><b>{meta['n_con_totalmin']}</b></p></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-box'><h3>Con Minutaje</h3><p><b>{meta['n_con_minutaje']}</b></p></div>", unsafe_allow_html=True)

            if meta["n_discrepancias"] and meta["n_discrepancias"] > 0:
                mapd_txt = f"{meta['mapd_discrepancia']:.2%}" if meta["mapd_discrepancia"] is not None else "N/D"
                st.info(f"Se detectaron **{meta['n_discrepancias']}** filas donde `Total Min` difiere de `minutaje×cantidad` por encima de la tolerancia ({tol_ui:.0%}). MAPD aproximado: **{mapd_txt}**. Revise el origen del dato o estandarice la fuente.")
            else:
                st.info("No se detectaron discrepancias relevantes entre `Total Min` y `minutaje×cantidad` por encima de la tolerancia.")

        if df_proc.empty:
            st.info("No se encontraron registros válidos tras aplicar las reglas de negocio.")
        else:
            modelo_rf, scaler_rf = cargar_modelo_rf()
            if modelo_rf is None:
                st.info("Falta el modelo en 'models': coloque 'modelo_curva_rf_class_bal.joblib' y 'scaler_X_rf_class_bal.joblib'.")
            else:
                X = df_proc[["cantidad", "minuto prenda", "minutos permanencia"]]
                try:
                    X_scaled = scaler_rf.transform(X)
                    pred_labels = modelo_rf.predict(X_scaled)
                except Exception:
                    st.info("No fue posible generar predicciones con el modelo RF cargado.")
                    pred_labels = None

                if pred_labels is not None:
                    df_res = df_proc.copy()
                    df_res["pred_categoria"] = pred_labels

                    st.markdown("Vista de registros clasificados (coherentes con la fórmula):")
                    st.dataframe(df_res.head(30), use_container_width=True)

                    conteo = df_res["pred_categoria"].value_counts().reindex(["Baja","Media","Alta"]).fillna(0).astype(int)

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='metric-box'><h3>Registros en Baja</h3><p><b>{conteo.get('Baja',0)}</b></p></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-box'><h3>Registros en Media</h3><p><b>{conteo.get('Media',0)}</b></p></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='metric-box'><h3>Registros en Alta</h3><p><b>{conteo.get('Alta',0)}</b></p></div>", unsafe_allow_html=True)

                    st.markdown("Distribución de niveles de eficiencia previstos:")
                    dist_df = pd.DataFrame({"Nivel":["Baja","Media","Alta"],
                                            "Cantidad":[conteo.get("Baja",0), conteo.get("Media",0), conteo.get("Alta",0)]}).set_index("Nivel")
                    st.bar_chart(dist_df)

                    st.markdown("Curva de aprendizaje estimada (eficiencia promedio móvil):")
                    curva = df_res.sort_values("minutos permanencia") if df_res["minutos permanencia"].nunique()>1 else df_res.sort_index()
                    ventana = max(5, len(curva)//15)
                    curva["ef_media_movil"] = curva["eficiencia_pct"].rolling(window=ventana, min_periods=3).mean()
                    curva_plot = curva[["ef_media_movil"]].dropna()
                    if not curva_plot.empty:
                        curva_plot.index.name = "progreso"
                        st.line_chart(curva_plot)
                        st.caption("La curva suavizada muestra la tendencia de eficiencia. Un incremento progresivo refleja aprendizaje; estabilidad alta indica consolidación.")
                    else:
                        st.caption("No se pudo calcular una curva suavizada por escasez de datos válidos.")

                    st.info("El módulo aplica la fórmula oficial (Total Min / Min Trab × 100), valida coherencia del archivo y clasifica cada registro en Baja/Media/Alta para leer la curva de aprendizaje.")

# ======================================
# 3) COMPARACIÓN DE MODELOS
# ======================================
elif opcion == "Comparación de modelos":
    st.subheader("Comparativa de modelos de clasificación")
    dfm = construir_tabla_metricas().set_index("Modelo")
    st.dataframe(dfm.style.format("{:.4f}"), use_container_width=True)
    st.info("La comparación respalda la selección de RF como modelo principal; SVM y ANN sirven como referencias técnicas.")

# ======================================
# 4) MAPAS DE CALOR
# ======================================
elif opcion == "Mapas de calor":
    st.subheader("Mapas de calor y densidad")
    base_figuras = os.path.join(BASE_DIR, "figuras")
    carpetas = [
        os.path.join(base_figuras, "graficas_calor"),
        os.path.join(base_figuras, "exploratorias")
    ]
    mostrado = False
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            for nombre in os.listdir(carpeta):
                if nombre.lower().endswith((".png",".jpg",".jpeg")):
                    st.image(os.path.join(carpeta, nombre), use_container_width=True)
                    mostrado = True
    if not mostrado:
        st.info("No se encontraron mapas de calor. Genere las figuras antes de esta visualización.")

# ======================================
# 5) ANÁLISIS AVANZADO
# ======================================
elif opcion == "Análisis avanzado":
    st.subheader("Gráficas avanzadas de comportamiento")
    base_figuras = os.path.join(BASE_DIR, "figuras")
    carpetas = [
        os.path.join(base_figuras, "graficas_avanzadas"),
        os.path.join(base_figuras, "ann")
    ]
    mostrado = False
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            for nombre in os.listdir(carpeta):
                if nombre.lower().endswith((".png",".jpg",".jpeg")):
                    st.image(os.path.join(carpeta, nombre), use_container_width=True)
                    mostrado = True
    if not mostrado:
        st.info("No se encontraron gráficas avanzadas. Genere las figuras antes de esta visualización.")

# ======================================
# 6) INFORMACIÓN DEL PROYECTO
# ======================================
elif opcion == "Información del proyecto":
    st.subheader("Información del Proyecto")
    st.markdown("""
**Proyecto:** Modelo predictivo aplicando Machine Learning para la identificación de la curva de aprendizaje en la producción textil.  
**Cliente:** Topitop S.A.  
**Equipo:** Hector Agustín Garcia Cortez - Jorge Hiro Chung Quispe  
**Institución:** Universidad Privada del Norte – Ingeniería de Sistemas Computacionales – 2025  
**Metodología:** CRISP–DM  
**Entorno:** Python 3.10 – Visual Studio Code  
**Bibliotecas:** pandas, scikit-learn, TensorFlow/Keras, seaborn, matplotlib, streamlit  
""")
    st.info("El panel aplica la fórmula oficial (Total Min / Min Trab × 100), valida consistencia de archivo y clasifica automáticamente cada registro en Baja/Media/Alta para monitorear la curva de aprendizaje.")
