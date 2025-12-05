import os
import numpy as np
import pandas as pd
import joblib

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

def cargar_datos_default():
    """
    Carga las hojas L72 y L79 del Excel base.
    Es la misma lógica que en el dashboard, pero sin Streamlit.
    """
    if not os.path.exists(DATA_PATH_DEFAULT):
        raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH_DEFAULT}")

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

    Devuelve un DataFrame con:
      cantidad, minutaje, min_trab, minutos_producidos,
      eficiencia_pct, eficiencia, categoria (Baja/Media/Alta)
      y, si existen, columnas de fecha y prenda.
    """
    df = df_raw.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    if "tipo" in df.columns:
        df = df[df["tipo"].astype(str).str.contains("salida", case=False, na=False)]

    df = df.loc[:, ~df.columns.duplicated()]

    cand_cant = [c for c in df.columns if "cant" in c]
    if not cand_cant:
        return pd.DataFrame()
    col_cant = cand_cant[0]

    cand_trab = [c for c in df.columns if "min trab" in c or "min_trab" in c or "perman" in c]
    if not cand_trab:
        return pd.DataFrame()
    col_min_trab = cand_trab[0]

    cand_minutaje = [
        c
        for c in df.columns
        if "mix" in c or "minutaje" in c or ("minuto" in c and "prenda" in c)
    ]
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

    bins = [0, 70, 85, 120]
    labels = ["Baja", "Media", "Alta"]
    df_feat["categoria"] = pd.cut(
        df_feat["eficiencia_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    col_fecha = [c for c in df.columns if "fecha" in c]
    if col_fecha:
        fecha_series = pd.to_datetime(
            df[col_fecha[0]],
            errors="coerce",
            dayfirst=True,
        )
        df_feat["fecha"] = fecha_series.reindex(df_feat.index)

    col_prenda = [c for c in df.columns if "prenda" in c or "estilo" in c or "modelo" in c]
    if col_prenda:
        prenda_series = df[col_prenda[0]].astype(str)
        df_feat["prenda"] = prenda_series.reindex(df_feat.index).fillna("Sin prenda")

    return df_feat

MODELOS = {
    "Random Forest": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_rf_class_bal.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_rf_class_bal.joblib"),
    },
    "Regresión Logística": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_log_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_log_class.joblib"),
    },
    "SVM": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_svm_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_svm_class.joblib"),
    },
    "Red Neuronal (ANN)": {
        "model_path": os.path.join(MODELS_DIR, "modelo_curva_ann_class.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "scaler_X_ann_class.joblib"),
    },
}


def cargar_modelo_y_scaler(nombre_modelo_mostrado: str):
    """
    Carga modelo y scaler desde MODELOS.
    Devuelve (modelo, scaler) o (None, None) si no existen.
    """
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
