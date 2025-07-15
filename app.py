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

st.sidebar.header("Ingrese los datos del paciente")
entrada = {
    'Embarazos': st.sidebar.number_input("Número de embarazos", 0, 20, 1),
    'Glucosa': st.sidebar.number_input("Nivel de glucosa", 0.0, 300.0, 120.0),
    'Presion': st.sidebar.number_input("Presión arterial", 0.0, 180.0, 70.0),
    'Pliegue': st.sidebar.number_input("Grosor del pliegue cutáneo", 0.0, 99.0, 20.0),
    'Insulina': st.sidebar.number_input("Nivel de insulina", 0.0, 900.0, 79.0),
    'IMC': st.sidebar.number_input("IMC", 0.0, 70.0, 30.0),
    'Pedigri': st.sidebar.number_input("Pedigrí de diabetes", 0.0, 2.5, 0.5),
    'Edad': st.sidebar.number_input("Edad", 1, 120, 33)
}

if st.sidebar.button("Predecir"):
    input_df = pd.DataFrame([entrada])
    pred = mejor_modelo.predict(input_df)[0]
    prob = mejor_modelo.predict_proba(input_df)[0][1]
    resultado = "Positivo para diabetes tipo 2" if pred == 1 else "Negativo para diabetes tipo 2"
    st.success(f"Resultado: {resultado}")
    st.info(f"Probabilidad estimada: {prob:.2f}")

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
