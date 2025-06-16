# app.py
import streamlit as st
from transformers import pipeline

id2label = {
    "LABEL_0": "Male",
    "LABEL_1": "Female"
}

# Cargar el pipeline de Hugging Face
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="hzerrweckk0101/gender-text-predictor")

# Cargar modelo
classifier = load_model()

# Interfaz de usuario
st.title("Gender Text Predictor")
st.write("Escribe un texto para predecir el género de quien lo escribió.")

# Entrada de texto
user_input = st.text_area("Introduce tu texto aquí:", height=200)

# Botón para predecir
if st.button("Predecir género"):
    if user_input.strip() != "":
        prediction = classifier(user_input)
        label_raw = prediction[0]['label']
        label = id2label.get(label_raw, label_raw)  # traducir si existe en el diccionario
        score = prediction[0]['score']

        st.success(f"Predicción: **{label}** con confianza de **{score:.2%}**")
    else:
        st.warning("Por favor, introduce un texto para predecir.")
