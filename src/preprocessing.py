import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_EXCEL = os.path.join(BASE_DIR, "data", "2 Salida de prendas.xlsx")
RUTA_SALIDA_CSV = os.path.join(BASE_DIR, "data", "processed_topitop.csv")

print(f"Leyendo archivo de origen: {RUTA_EXCEL}")
print(f"El CSV procesado se guardará en: {RUTA_SALIDA_CSV}")


def _find_col(df: pd.DataFrame, pred) -> str | None:
    """
    Devuelve el nombre de la primera columna que cumpla el predicado `pred`,
    o None si no encuentra ninguna.
    """
    hits = [c for c in df.columns if pred(c)]
    return hits[0] if hits else None


def cargar_y_unir_hojas(path_excel: str) -> pd.DataFrame:
    """
    Carga todas las hojas del Excel (o solo L72/L79 si quieres restringir)
    y las concatena en un único DataFrame.
    """
    if not os.path.exists(path_excel):
        raise FileNotFoundError(f"No se encontró el Excel en: {path_excel}")

    xls = pd.ExcelFile(path_excel)
    hojas = xls.sheet_names              # si quieres solo L72/L79: ["L72", "L79"]
    print(f"Hojas encontradas: {hojas}")

    df_list = []
    for hoja in hojas:
        tmp = xls.parse(hoja)
        df_list.append(tmp)
        print(f"  - Hoja {hoja}: {tmp.shape[0]} filas, {tmp.shape[1]} columnas")

    df = pd.concat(df_list, ignore_index=True)
    print(f"DataFrame combinado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def preprocesar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la misma lógica que usa el dashboard:

      - Normaliza nombres de columnas
      - Filtra solo registros TIPO 'salida'
      - Calcula:
          minutos_producidos = minutaje * cantidad
          eficiencia_pct     = (minutos_producidos / min_trab) * 100
          eficiencia         = eficiencia_pct / 100
      - Genera categoría (Baja, Media, Alta) en función de eficiencia_pct
    """
    df = df_raw.copy()

    # Normalizar nombres
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    # Solo tipo "salida" si existe la columna tipo
    if "tipo" in df.columns:
        antes = len(df)
        df = df[df["tipo"].astype(str).str.contains("salida", case=False, na=False)]
        print(f"Filtrado por TIPO=salida: {antes} -> {len(df)} filas")

    # Quitar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # ---------- Buscar columnas clave ----------
    # cantidad
    col_cant = _find_col(df, lambda c: "cant" in c)
    if not col_cant:
        raise RuntimeError("No se encontró columna de 'cantidad' (ej. contiene 'cant').")

    # minutos de trabajo (min_trab / permanencia / min trab)
    col_min_trab = _find_col(
        df,
        lambda c: "min trab" in c or "min_trab" in c or "perman" in c,
    )
    if not col_min_trab:
        raise RuntimeError(
            "No se encontró columna de minutos de trabajo (ej. contiene 'min trab', 'min_trab' o 'perman')."
        )

    # minutaje por prenda (minuto prenda / mix / minutaje / total minutos)
    col_minutaje = _find_col(
        df,
        lambda c: "mix" in c
        or "minutaje" in c
        or ("minuto" in c and "prenda" in c)
    )

    if col_minutaje:
        # Tenemos directamente el minutaje
        df["minutaje"] = pd.to_numeric(df[col_minutaje], errors="coerce")
        print(f"Usando columna '{col_minutaje}' como minutaje por prenda.")
    else:
        # Buscar total de minutos y dividir entre cantidad
        col_total_min = _find_col(df, lambda c: "total" in c and "min" in c)
        if not col_total_min:
            raise RuntimeError(
                "No se encontró columna de minutaje ni total de minutos para derivarlo."
            )
        print(
            f"No se encontró minutaje directo. "
            f"Usando '{col_total_min} / {col_cant}' para calcular minutaje."
        )
        df["minutaje"] = (
            pd.to_numeric(df[col_total_min], errors="coerce")
            / pd.to_numeric(df[col_cant], errors="coerce")
        )

    # ---------- Construir dataframe de features ----------
    df_feat = df[[col_cant, "minutaje", col_min_trab]].copy()
    df_feat.columns = ["cantidad", "minutaje", "min_trab"]
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]
    df_feat = df_feat.apply(pd.to_numeric, errors="coerce")

    # Filtrar valores no válidos
    antes_validos = len(df_feat)
    df_feat = df_feat[
        (df_feat["cantidad"] > 0)
        & (df_feat["minutaje"] > 0)
        & (df_feat["min_trab"] > 0)
    ].copy()
    print(f"Filtrado de registros con valores <= 0: {antes_validos} -> {len(df_feat)} filas")

    if df_feat.empty:
        raise RuntimeError("Tras el filtrado no quedaron filas válidas.")

    # ---------- Cálculos finales ----------
    df_feat["minutos_producidos"] = df_feat["minutaje"] * df_feat["cantidad"]
    df_feat["eficiencia_pct"] = (df_feat["minutos_producidos"] / df_feat["min_trab"]) * 100.0
    df_feat["eficiencia"] = df_feat["eficiencia_pct"] / 100.0

    # Limitar eficiencias a un rango razonable (0–120%)
    antes_rango = len(df_feat)
    df_feat = df_feat[
        (df_feat["eficiencia_pct"] >= 0) & (df_feat["eficiencia_pct"] <= 120)
    ].copy()
    print(f"Filtrado por rango de eficiencia (0–120%): {antes_rango} -> {len(df_feat)} filas")

    # Categorizar (igual que en el dashboard)
    bins = [0, 70, 85, 100]
    labels = ["Baja", "Media", "Alta"]
    df_feat["categoria"] = pd.cut(
        df_feat["eficiencia_pct"], bins=bins, labels=labels, include_lowest=True
    )

    # Fecha (si existe)
    col_fecha = _find_col(df, lambda c: "fecha" in c)
    if col_fecha:
        df_feat["fecha"] = pd.to_datetime(
            df[col_fecha], errors="coerce", dayfirst=True
        )

    # Prenda / modelo / estilo (si existe)
    col_prenda = _find_col(
        df, lambda c: "prenda" in c or "estilo" in c or "modelo" in c
    )
    if col_prenda:
        df_feat["prenda"] = df[col_prenda].astype(str).fillna("Sin prenda")

    print("Resumen de categorías:")
    print(df_feat["categoria"].value_counts(dropna=False))

    return df_feat


def main():
    # 1) Cargar Excel
    df_raw = cargar_y_unir_hojas(RUTA_EXCEL)

    # 2) Preprocesar
    df_proc = preprocesar(df_raw)

    # 3) Guardar CSV
    df_proc.to_csv(RUTA_SALIDA_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Preprocesamiento completo. CSV guardado en:\n{RUTA_SALIDA_CSV}")
    print(f"Shape final: {df_proc.shape[0]} filas x {df_proc.shape[1]} columnas")


if __name__ == "__main__":
    main()
