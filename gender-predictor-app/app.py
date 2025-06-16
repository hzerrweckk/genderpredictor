import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Reemplaza con tu usuario si es necesario
MODEL_NAME = "tu_usuario/gender-text-predictor"

# Diccionario para convertir IDs en etiquetas legibles
id2label = {0: "Male", 1: "Female"}

# Cargar modelo y tokenizer desde Hugging Face Hub
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Función de predicción
def predict_gender(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()
    gender = id2label.get(pred, f"LABEL_{pred}")
    return gender, confidence

# Interfaz con Streamlit
st.title("🧠 Gender Prediction from Writing")
st.write("Enter a text to predict the author's gender based on writing style.")

user_input = st.text_area("Enter text here:", height=200)

if st.button("Predict"):
    if user_input.strip():
        gender, confidence = predict_gender(user_input)
        st.success(f"Predicted Gender: {gender} ({confidence:.2%} confidence)")
    else:
        st.warning("Please enter some text first.")
