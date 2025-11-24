# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------------------------------
# TÍTULO PRINCIPAL
# -----------------------------------------------------
st.title("🌸 Iris Species Classification – Dashboard Profesional")


# -----------------------------------------------------
# CARGA DEL DATASET
# -----------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"], errors="ignore")

numeric_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

le = LabelEncoder()
df["Species_cod"] = le.fit_transform(df["Species"])


# -----------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------
X = df[numeric_cols]
y = df["Species_cod"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------------------------
# PCA (3 Componentes para visualización 3D)
# -----------------------------------------------------
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)

df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
df_pca["Species"] = df["Species"]


# -----------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)


# -----------------------------------------------------
# SIDEBAR – NAVEGACIÓN
# -----------------------------------------------------
menu = st.sidebar.radio(
    "Navegación",
    ["📘 Metodología", "📊 Métricas del Modelo", "📈 Visualizaciones",
     "🧭 PCA 3D (Requisito)", "🔮 Predicción del Usuario"]
)


# -----------------------------------------------------
# 📘 SECCIÓN METODOLOGÍA
# -----------------------------------------------------
if menu == "📘 Metodología":
    st.header("📘 Metodología del Proyecto")

    st.markdown("""
### ✔ 1. Comprensión del Dataset
Se trabaja con **Iris.csv**, compuesto por 150 muestras de 3 especies:
- Iris-setosa  
- Iris-versicolor  
- Iris-virginica  

Cada una con 4 características numéricas.

---

### ✔ 2. Preprocesamiento
- Eliminación de la columna `Id`.
- Estandarización de características con `StandardScaler`.
- Codificación numérica de especies mediante `LabelEncoder`.

---

### ✔ 3. Modelado
Se utiliza un **Random Forest** con:
- 200 árboles  
- Random state 42  
- Entrenamiento sobre datos escalados  

---

### ✔ 4. Validación
Se evalúa con métricas:
- Accuracy  
- Precision  
- Recall  
- F1  
- Matriz de Confusión  

---

### ✔ 5. Visualización
Incluye:
- Histogramas  
- 3D Scatter clásico  
- PCA 3D  
- Posición del nuevo punto ingresado  

---

### ✔ 6. Interfaz
Aplicación desarrollada en Streamlit, navegable por secciones.
""")


# -----------------------------------------------------
# 📊 SECCIÓN MÉTRICAS DEL MODELO
# -----------------------------------------------------
elif menu == "📊 Métricas del Modelo":
    st.header("📊 Métricas del Modelo")

    y_pred = model.predict(X_test_scaled)

    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.4f}")

    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    st.pyplot(fig)


# -----------------------------------------------------
# 📈 SECCIÓN VISUALIZACIONES BÁSICAS
# -----------------------------------------------------
elif menu == "📈 Visualizaciones":
    st.header("📈 Visualizaciones del Dataset")

    # Histogramas
    st.subheader("Histogramas")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=15, edgecolor="black")
        axes[i].set_title(f"Histograma de {col}")

    st.pyplot(fig)

    # Scatter 3D tradicional
    st.subheader("Gráfico 3D del Dataset")
    fig3d = px.scatter_3d(
        df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        z="SepalLengthCm",
        color="Species",
        title="Dataset Iris – 3D Scatter Plot"
    )
    st.plotly_chart(fig3d)


# -----------------------------------------------------
# 🧭 PCA 3D
# -----------------------------------------------------
elif menu == "🧭 PCA 3D":
    st.header("🧭 Visualización PCA 3D")

    fig_pca = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Species",
        title="PCA 3D del Dataset Iris"
    )
    st.plotly_chart(fig_pca)


# -----------------------------------------------------
# 🔮 SECCIÓN PREDICCIÓN DEL USUARIO
# -----------------------------------------------------
elif menu == "🔮 Predicción del Usuario":
    st.header("🔮 Predicción de Especie")

    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

    if st.button("Predecir"):
        new_data = np.array([[sl, sw, pl, pw]])
        new_scaled = scaler.transform(new_data)
        pred = model.predict(new_scaled)[0]
        pred_name = le.inverse_transform([pred])[0]

        st.success(f"🌼 La especie predicha es: **{pred_name}**")

        # PCA del nuevo punto
        new_pca = pca.transform(new_data)
        df_new = pd.DataFrame({
            "PC1": [new_pca[0, 0]],
            "PC2": [new_pca[0, 1]],
            "PC3": [new_pca[0, 2]],
            "Species": ["Nuevo Punto"]
        })

        df_all = pd.concat([df_pca, df_new])

        fig_pred = px.scatter_3d(
            df_all,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Species",
            symbol="Species",
            title="Posición del Nuevo Punto en PCA 3D"
        )
        st.plotly_chart(fig_pred)
