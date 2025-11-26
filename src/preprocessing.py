# preprocessing.py
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_EXCEL = os.path.join(BASE_DIR, "data", "2 Salida de prendas.xlsx")
RUTA_SALIDA_CSV = os.path.join(BASE_DIR, "data", "processed_topitop.csv")

print(f"Leyendo archivo: {RUTA_EXCEL}")

def _find_col(df, pred):
    hits = [c for c in df.columns if pred(c)]
    return hits[0] if hits else None

try:
    # 1) Carga de todas las hojas disponibles (L72/L79 u otras si las hubiera)
    xls = pd.ExcelFile(RUTA_EXCEL)
    hojas = xls.sheet_names
    df_list = []
    for hoja in hojas:
        tmp = xls.parse(hoja)
        df_list.append(tmp)
        print(f"Hoja {hoja} cargada: {tmp.shape}")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Dataset combinado: {df.shape}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("'", "")
        .str.replace('"', "")
    )

    # Filtrar sólo registros tipo "Salida" (si la columna existe)
    if "tipo" in df.columns:
        before = len(df)
        df = df[df["tipo"].str.contains("salida", case=False, na=False)]
        print(f"Filtro tipo='Salida': {before} -> {len(df)} registros")

    # Localizar columnas según plantilla (tolerante a nombres)
    col_linea  = _find_col(df, lambda c: "línea" in c or "linea" in c)
    col_estilo = _find_col(df, lambda c: "estilo" in c)
    col_op     = _find_col(df, lambda c: c == "op" or "orden de producción" in c or "orden de produccion" in c or c.startswith("op"))
    #  minutaje estilo / "mix" o similar
    col_minut  = _find_col(df, lambda c: ("mix" in c) or ("minut" in c and "perman" not in c and "prod" not in c))
    #  cantidad
    col_cant   = _find_col(df, lambda c: "cant" in c)
    #  total min (minutos producidos)
    col_total  = _find_col(df, lambda c: ("total" in c and "min" in c) or ("min" in c and "prod" in c))
    #  min trab (minutos permanencia)
    col_mtrab  = _find_col(df, lambda c: "min trab" in c or "perman" in c or c == "n")
    # fecha (opcional)
    col_fecha  = _find_col(df, lambda c: "fecha" in c)

    # Chequeo mínimo: necesitamos cantidad y min_trab, y además total_min o minutaje
    if (col_cant is None) or (col_mtrab is None) or (col_total is None and col_minut is None):
        faltan = {
            "cantidad": col_cant is None,
            "min_trab": col_mtrab is None,
            "total_min/minutaje": (col_total is None and col_minut is None),
        }
        raise ValueError(f"Faltan columnas clave para la plantilla: {faltan}")

    # 5) Selección y tipos numéricos
    keep_cols = [c for c in [col_linea, col_estilo, col_op, col_minut, col_fecha, col_cant, col_total, col_mtrab] if c]
    df = df[keep_cols].copy()

    num_cols = [x for x in [col_minut, col_cant, col_total, col_mtrab] if x]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 6) Renombrar estándar
    rename = {}
    if col_linea:  rename[col_linea]  = "linea"
    if col_estilo: rename[col_estilo] = "estilo"
    if col_op:     rename[col_op]     = "op"
    if col_minut:  rename[col_minut]  = "minutaje"          
    rename[col_cant]  = "cantidad"
    if col_total:  rename[col_total]  = "total_min"         
    rename[col_mtrab] = "min_trab"                           
    if col_fecha:  rename[col_fecha]  = "fecha"
    df = df.rename(columns=rename)

    if "total_min" in df.columns:
        df["minutos_producidos"] = df["total_min"]
        if "minutaje" in df.columns:
            calc = df["minutaje"] * df["cantidad"]
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_ratio = np.abs(df["minutos_producidos"] - calc) / np.where(df["minutos_producidos"] != 0, df["minutos_producidos"], np.nan)
            n_disc = int((diff_ratio > 0.05).fillna(False).sum())
            mapd   = float(np.nanmean(diff_ratio)) if np.isfinite(np.nanmean(diff_ratio)) else np.nan
            print(f"Discrepancias Total Min vs (Minutaje×Cantidad) >5%: {n_disc} | MAPD≈ {mapd if not np.isnan(mapd) else 'N/D'}")
            # Si total_min falta o es 0, reemplazar por el calculado
            df.loc[df["minutos_producidos"].isna() | (df["minutos_producidos"] <= 0), "minutos_producidos"] = calc
    else:
        df["minutos_producidos"] = df["minutaje"] * df["cantidad"]

    before_valid = len(df)
    df = df[
        (df["cantidad"] > 0) &
        (df["min_trab"] > 0) &
        (df["minutos_producidos"] > 0)
    ].copy()
    print(f"Filtrado de inválidos (<=0/NaN): {before_valid} -> {len(df)}")

    if df.empty:
        raise ValueError("Sin registros válidos luego del filtrado básico.")

    # Eficiencia según fórmula oficial
    df["eficiencia_pct"] = (df["minutos_producidos"] / df["min_trab"]) * 100.0
    df["eficiencia"] = df["eficiencia_pct"] / 100.0

    #  Outliers razonables de eficiencia
    before_out = len(df)
    df = df[(df["eficiencia_pct"] >= 0) & (df["eficiencia_pct"] <= 120)].copy()
    print(f"Filtro outliers de eficiencia [0%,120%]: {before_out} -> {len(df)}")

    # Clasificación en Baja/Media/Alta (para modelos de clasificación)
    bins = [0, 70, 85, 100]
    labels = ["Baja", "Media", "Alta"]
    df["categoria"] = pd.cut(df["eficiencia_pct"], bins=bins, labels=labels, include_lowest=True)

    # Orden de columnas “amigable”
    ordered = ["linea", "estilo", "op", "minutaje", "fecha",
               "cantidad", "minutos_producidos", "min_trab",
               "eficiencia_pct", "eficiencia", "categoria"]
    df = df[[c for c in ordered if c in df.columns]].copy()

    # Resumen
    print("Resumen de columnas finales:", list(df.columns))
    print("Primeras filas:")
    print(df.head(10))

    # Guardado del dataset limpio
    try:
        df.to_csv(RUTA_SALIDA_CSV, index=False, encoding="utf-8")
        print(f"Dataset limpio guardado en: {RUTA_SALIDA_CSV}")
    except Exception as e_save:
        print(f"No se pudo guardar el CSV de salida: {e_save}")

    print(f"Datos limpios listos: {df.shape[0]} filas.")

except Exception as e:
    print(f"Error al cargar y preparar datos: {e}")
