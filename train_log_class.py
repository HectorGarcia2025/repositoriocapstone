# ======================================
# 📘 Regresión Logística (Clasificación Balanceada)
# Curva de Aprendizaje Textil - Topitop
# ======================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import resample

# ==========================
# 1️⃣ CARGA Y LIMPIEZA DE DATOS
# ==========================
print("📂 Leyendo datos...")
ruta_excel = r"C:\Users\HECTOR GARCIA\Desktop\Capstone\Capstone\modelo_textil_ml\data\2 Salida de prendas.xlsx"

df_l72 = pd.read_excel(ruta_excel, sheet_name="L72")
df_l79 = pd.read_excel(ruta_excel, sheet_name="L79")
df = pd.concat([df_l72, df_l79], ignore_index=True)
df.columns = df.columns.str.strip().str.lower().str.replace("'", "").str.replace('"', "")

if "tipo" in df.columns:
    df = df[df["tipo"].str.contains("salida", case=False, na=False)]

cols = {
    "cantidad": [c for c in df.columns if "cant" in c][0],
    "minuto prenda": [c for c in df.columns if "minuto" in c and "prenda" in c][0],
    "minutos permanencia": [c for c in df.columns if "perman" in c][0],
    "eficiencia": [c for c in df.columns if "efic" in c][0],
}
df = df[[cols["cantidad"], cols["minuto prenda"], cols["minutos permanencia"], cols["eficiencia"]]]
df.columns = ["cantidad", "minuto prenda", "minutos permanencia", "eficiencia"]

df = df.apply(pd.to_numeric, errors="coerce").dropna()
df["eficiencia"] = df["eficiencia"] * 100

bins = [0, 70, 85, 100]
labels = ["Baja", "Media", "Alta"]
df["categoria"] = pd.cut(df["eficiencia"], bins=bins, labels=labels, include_lowest=True)
df = df.dropna(subset=["categoria"])

print("\nDistribución original:")
print(df["categoria"].value_counts())

# ==========================
# 2️⃣ BALANCEO DE CLASES
# ==========================
df_baja = df[df["categoria"] == "Baja"]
df_media = df[df["categoria"] == "Media"]
df_alta = df[df["categoria"] == "Alta"]

df_media_up = resample(df_media, replace=True, n_samples=len(df_baja), random_state=42)
df_alta_up = resample(df_alta, replace=True, n_samples=len(df_baja), random_state=42)

df_bal = pd.concat([df_baja, df_media_up, df_alta_up])
df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDistribución balanceada:")
print(df_bal["categoria"].value_counts())

# ==========================
# 3️⃣ ESCALADO Y DIVISIÓN
# ==========================
X = df_bal[["cantidad", "minuto prenda", "minutos permanencia"]]
y = df_bal["categoria"]

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 4️⃣ ENTRENAMIENTO
# ==========================
modelo = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
modelo.fit(X_train, y_train)

# ==========================
# 5️⃣ EVALUACIÓN
# ==========================
y_pred = modelo.predict(X_test)
y_bin = label_binarize(y_test, classes=["Baja", "Media", "Alta"])
y_proba = modelo.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

try:
    auc = roc_auc_score(y_bin, y_proba, multi_class="ovr")
except:
    auc = np.nan

cm = confusion_matrix(y_test, y_pred, labels=["Baja", "Media", "Alta"])

print("\n📊 MÉTRICAS REGRESIÓN LOGÍSTICA (Balanceado)")
print(f"Accuracy: {acc:.4f}")
print(f"Precisión: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================
# 6️⃣ GUARDADO
# ==========================
os.makedirs("../modelo_textil_ml/models", exist_ok=True)
joblib.dump(modelo, "../modelo_textil_ml/models/modelo_curva_log_class.joblib")
joblib.dump(scaler_X, "../modelo_textil_ml/models/scaler_X_log_class.joblib")

resultados = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc,
    "confusion_matrix": cm.tolist(),
}
joblib.dump(resultados, "../modelo_textil_ml/models/metricas_log_class.joblib")

print("\n✅ Modelo de Regresión Logística balanceado guardado correctamente.")
