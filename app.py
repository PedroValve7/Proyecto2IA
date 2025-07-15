# app.py - Aplicación web de predicción de diabetes con Streamlit

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columnas = ['Embarazos', 'Glucosa', 'Presion', 'Pliegue', 'Insulina', 'IMC', 'Pedigri', 'Edad', 'Clase']
    df = pd.read_csv(url, names=columnas)
    for col in ['Glucosa', 'Presion', 'Pliegue', 'Insulina', 'IMC']:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())
    return df

df = cargar_datos()
X = df.drop('Clase', axis=1)
y = df['Clase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

resultados = {}
y_scores = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test)
    resultados[nombre] = {
        'modelo': modelo,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }
    y_scores[nombre] = y_prob

mejor_modelo_nombre = max(resultados, key=lambda k: resultados[k]['f1'])
mejor_modelo = resultados[mejor_modelo_nombre]['modelo']

st.title("Predicción de Diabetes Tipo 2")
st.subheader(f"Mejor modelo: {mejor_modelo_nombre}")

# Ejemplos precargados
ejemplos = {
    "Ejemplo 1 - Bajo riesgo": {
        'Embarazos': 1,
        'Glucosa': 85.0,
        'Presion': 66.0,
        'Pliegue': 29.0,
        'Insulina': 79.0,
        'IMC': 26.6,
        'Pedigri': 0.351,
        'Edad': 31
    },
    "Ejemplo 2 - Alto riesgo": {
        'Embarazos': 5,
        'Glucosa': 140.0,
        'Presion': 80.0,
        'Pliegue': 35.0,
        'Insulina': 130.0,
        'IMC': 40.5,
        'Pedigri': 0.7,
        'Edad': 45
    }
}

# Mostrar predicciones para ejemplos precargados
st.subheader("Predicciones para ejemplos precargados")
for nombre, datos in ejemplos.items():
    df_input = pd.DataFrame([datos])
    pred = mejor_modelo.predict(df_input)[0]
    prob = mejor_modelo.predict_proba(df_input)[0][1]
    resultado = "🟢 Negativo para diabetes tipo 2" if pred == 0 else "🔴 Positivo para diabetes tipo 2"
    color = "green" if pred == 0 else "red"
    st.markdown(f"**{nombre}:** <span style='color:{color}; font-weight:bold'>{resultado}</span>", unsafe_allow_html=True)
    st.markdown(f"> Probabilidad estimada: **{prob:.2f}**")
    st.write("---")

st.sidebar.header("Ingrese los datos del paciente")

# Selector para cargar ejemplos
cargar_ejemplo = st.sidebar.selectbox("Cargar ejemplo precargado", ["Ninguno"] + list(ejemplos.keys()))

# Inicializar session_state para inputs si no existe
def inicializar_estado():
    defaults = {
        "Embarazos": 1,
        "Glucosa": 120.0,
        "Presion": 70.0,
        "Pliegue": 20.0,
        "Insulina": 79.0,
        "IMC": 30.0,
        "Pedigri": 0.5,
        "Edad": 33
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

inicializar_estado()

# Actualizar session_state con valores del ejemplo seleccionado
if cargar_ejemplo != "Ninguno":
    ejemplo = ejemplos[cargar_ejemplo]
    for key, value in ejemplo.items():
        if st.session_state[key] != value:
            st.session_state[key] = value

# Mostrar inputs con valores desde session_state para poder editarlos
entrada = {
    'Embarazos': st.sidebar.number_input("Número de embarazos", 0, 20, st.session_state.Embarazos, key='Embarazos'),
    'Glucosa': st.sidebar.number_input("Nivel de glucosa", 0.0, 300.0, st.session_state.Glucosa, key='Glucosa'),
    'Presion': st.sidebar.number_input("Presión arterial", 0.0, 180.0, st.session_state.Presion, key='Presion'),
    'Pliegue': st.sidebar.number_input("Grosor del pliegue cutáneo", 0.0, 99.0, st.session_state.Pliegue, key='Pliegue'),
    'Insulina': st.sidebar.number_input("Nivel de insulina", 0.0, 900.0, st.session_state.Insulina, key='Insulina'),
    'IMC': st.sidebar.number_input("IMC", 0.0, 70.0, st.session_state.IMC, key='IMC'),
    'Pedigri': st.sidebar.number_input("Pedigrí de diabetes", 0.0, 2.5, st.session_state.Pedigri, key='Pedigri'),
    'Edad': st.sidebar.number_input("Edad", 1, 120, st.session_state.Edad, key='Edad')
}

if st.sidebar.button("Predecir"):
    input_df = pd.DataFrame([entrada])
    pred = mejor_modelo.predict(input_df)[0]
    prob = mejor_modelo.predict_proba(input_df)[0][1]
    resultado = "🟢 Negativo para diabetes tipo 2" if pred == 0 else "🔴 Positivo para diabetes tipo 2"
    color = "green" if pred == 0 else "red"
    st.markdown(f"**Resultado:** <span style='color:{color}; font-weight:bold'>{resultado}</span>", unsafe_allow_html=True)
    st.markdown(f"Probabilidad estimada: **{prob:.2f}**")

st.subheader("Comparación de métricas")
df_metricas = pd.DataFrame({
    nombre: [res['accuracy'], res['recall'], res['f1']] for nombre, res in resultados.items()
}, index=['Precisión', 'Recall', 'F1-score']).T
st.bar_chart(df_metricas)

st.subheader("Matrices de confusión")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, (nombre, res) in enumerate(resultados.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(nombre)
    axes[i].set_xlabel("Predicción")
    axes[i].set_ylabel("Real")
st.pyplot(fig)

st.subheader("Curvas ROC")
fig2, ax2 = plt.subplots(figsize=(8, 6))
for nombre in modelos:
    fpr, tpr, _ = roc_curve(y_test, y_scores[nombre])
    auc_score = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f"{nombre} (AUC={auc_score:.2f})")
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel("Falsos Positivos")
ax2.set_ylabel("Verdaderos Positivos")
ax2.set_title("Curvas ROC")
ax2.legend()
ax2.grid()
st.pyplot(fig2)
