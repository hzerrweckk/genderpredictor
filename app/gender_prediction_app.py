# gender_prediction_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Load tokenizer and model (fine-tuned on gender detection task)
MODEL_NAME = "hzerrweckk0101/gender-text-predictor" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Streamlit UI
st.title("Gender Prediction from Text")
st.write("Enter a text sample to predict the author's gender:")

user_input = st.text_area("Text Input", height=200)

if st.button("Predict"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0, pred_class].item()

        gender = "Male" if pred_class == 0 else "Female"
        st.success(f"Predicted Gender: {gender} ({confidence*100:.2f}% confidence)")
    else:
        st.warning("Please enter some text to predict.")
