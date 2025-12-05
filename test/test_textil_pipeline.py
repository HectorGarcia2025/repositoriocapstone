import os
import sys
import numpy as np
import pandas as pd
import pytest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core_textil import (
    DATA_PATH_DEFAULT,
    cargar_datos_default,
    preprocesar,
    cargar_modelo_y_scaler,
    MODELOS,
)

def test_carga_excel_existente():
    """Verifica que el archivo Excel base exista y se pueda cargar."""
    if not os.path.exists(DATA_PATH_DEFAULT):
        pytest.skip(f"El archivo {DATA_PATH_DEFAULT} no existe en este entorno.")
    df_raw = cargar_datos_default()
    assert not df_raw.empty, "El Excel de producción no debería estar vacío."


def test_preprocesamiento_valores_validos():
    """
    Verifica que tras el preprocesamiento:
      - haya registros
      - cantidad, minutaje, min_trab sean > 0
      - eficiencia_pct esté en el rango 0–120
      - no existan NaN en columnas críticas
    """
    if not os.path.exists(DATA_PATH_DEFAULT):
        pytest.skip("No se encuentra el Excel base, se omite la prueba.")

    df_raw = cargar_datos_default()
    df = preprocesar(df_raw)
    assert not df.empty, "Después del preprocesamiento no debería quedar un DataFrame vacío."

    for col in ["cantidad", "minutaje", "min_trab", "eficiencia_pct"]:
        assert col in df.columns, f"Falta la columna requerida: {col}"

    assert (df["cantidad"] > 0).all()
    assert (df["minutaje"] > 0).all()
    assert (df["min_trab"] > 0).all()
    assert (df["eficiencia_pct"] >= 0).all()
    assert (df["eficiencia_pct"] <= 120).all()
    assert not df[["cantidad", "minutaje", "min_trab", "eficiencia_pct"]].isna().any().any()

def test_formula_minutos_y_eficiencia():
    """
    Comprueba que las fórmulas:
      minutos_producidos = minutaje * cantidad
      eficiencia_pct    = (minutos_producidos / min_trab) * 100
    se cumplan numéricamente.
    """
    if not os.path.exists(DATA_PATH_DEFAULT):
        pytest.skip("No se encuentra el Excel base, se omite la prueba.")

    df_raw = cargar_datos_default()
    df = preprocesar(df_raw)

    assert "minutos_producidos" in df.columns, "Falta la columna minutos_producidos."
    assert "eficiencia_pct" in df.columns, "Falta la columna eficiencia_pct."

    muestra = df.sample(min(50, len(df)), random_state=42)

    esper_min = muestra["minutaje"] * muestra["cantidad"]
    assert np.allclose(
        muestra["minutos_producidos"].values,
        esper_min.values,
        rtol=1e-5,
        atol=1e-5,
    ), "minutos_producidos no coincide con minutaje * cantidad."

    esper_eff = (muestra["minutos_producidos"] / muestra["min_trab"]) * 100
    assert np.allclose(
        muestra["eficiencia_pct"].values,
        esper_eff.values,
        rtol=1e-5,
        atol=1e-5,
    ), "eficiencia_pct no coincide con la fórmula definida."



def test_etiquetado_baja_media_alta():
    """
    Verifica que las etiquetas Baja/Media/Alta se asignen según los cortes 0–70–85–120.
    """
    if not os.path.exists(DATA_PATH_DEFAULT):
        pytest.skip("No se encuentra el Excel base, se omite la prueba.")

    df_raw = cargar_datos_default()
    df = preprocesar(df_raw)
    assert "categoria" in df.columns, "Falta la columna categoria."

    categorias_validas = {"Baja", "Media", "Alta"}
    unicos = set(df["categoria"].dropna().astype(str).unique())
    assert unicos.issubset(categorias_validas), \
        f"Se encontraron categorías inesperadas: {unicos - categorias_validas}"

    mask_baja = df["eficiencia_pct"] < 70
    mask_media = (df["eficiencia_pct"] >= 70) & (df["eficiencia_pct"] < 85)
    mask_alta = (df["eficiencia_pct"] >= 85) & (df["eficiencia_pct"] <= 120)

    if mask_baja.any():
        assert (df.loc[mask_baja, "categoria"] == "Baja").all()
    if mask_media.any():
        assert (df.loc[mask_media, "categoria"] == "Media").all()
    if mask_alta.any():
        assert (df.loc[mask_alta, "categoria"] == "Alta").all()


@pytest.mark.parametrize("nombre_modelo", list(MODELOS.keys()))
def test_modelos_predicen_clases_validas(nombre_modelo):
    """
    Verifica que cada modelo (si existe el .joblib) pueda:
      - cargarse sin error
      - predecir clases en {Baja, Media, Alta}
      - y, si tiene predict_proba, que las probabilidades sumen ~1
    """
    if not os.path.exists(DATA_PATH_DEFAULT):
        pytest.skip("No se encuentra el Excel base, se omite la prueba.")

    modelo, scaler = cargar_modelo_y_scaler(nombre_modelo)
    if modelo is None:
        pytest.skip(f"No se encontró el modelo entrenado para {nombre_modelo}.")

    df_raw = cargar_datos_default()
    df = preprocesar(df_raw)
    if df.empty:
        pytest.skip("No hay datos válidos después del preprocesamiento.")

    X = df[["cantidad", "minutaje", "min_trab"]].head(30).copy()
    if scaler is not None:
        X_np = scaler.transform(X)
    else:
        X_np = X.values

    preds = modelo.predict(X_np)
    assert len(preds) == len(X), "El número de predicciones debe coincidir con el número de registros."

    clases_validas = {"Baja", "Media", "Alta"}
    unicos_pred = set(map(str, np.unique(preds)))
    assert unicos_pred.issubset(clases_validas), \
        f"Clases predichas inesperadas: {unicos_pred - clases_validas}"

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_np)
        assert proba.shape[0] == len(X)
        suma_filas = proba.sum(axis=1)
        assert np.allclose(suma_filas, 1.0, atol=1e-5), "Las probabilidades por fila deben sumar 1."
