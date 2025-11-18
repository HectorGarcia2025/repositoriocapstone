# ======================================
# 🤖 Red Neuronal Artificial (Clasificación Balanceada)
# Curva de Aprendizaje Textil - Topitop
# ======================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.3, random_state=42, stratify=y_cat
)

# ==========================
# 4️⃣ ARQUITECTURA DE LA RED
# ==========================
modelo = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")  # 3 clases: Baja, Media, Alta
])

modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\n🧠 Entrenando red neuronal...")
historial = modelo.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)

# ==========================
# 5️⃣ EVALUACIÓN
# ==========================
y_pred_proba = modelo.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

try:
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
except:
    auc = np.nan

cm = confusion_matrix(y_true, y_pred)

print("\n📊 MÉTRICAS RED NEURONAL (Balanceado)")
print(f"Accuracy: {acc:.4f}")
print(f"Precisión: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0))

# ==========================
# 6️⃣ GUARDADO
# ==========================

historial = modelo.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Al final del script:
import joblib, os



print("✅ Historial de entrenamiento ANN guardado.")
os.makedirs("../modelo_textil_ml/models", exist_ok=True)
modelo.save("../modelo_textil_ml/models/modelo_curva_ann_class.keras")
joblib.dump(scaler_X, "../modelo_textil_ml/models/scaler_X_ann_class.joblib")
joblib.dump(encoder, "../modelo_textil_ml/models/encoder_ann_class.joblib")
joblib.dump(historial.history, "../modelo_textil_ml/models/historial_ann_class.joblib")

resultados = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc,
    "confusion_matrix": cm.tolist(),
}
joblib.dump(resultados, "../modelo_textil_ml/models/metricas_ann_class.joblib")

print("\n✅ Modelo ANN de clasificación balanceado guardado correctamente.")
