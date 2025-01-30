# app.py

import streamlit as st
from PIL import Image
from utils.predict import load_model, predict_image
from pathlib import Path
import keras

# Load the trained model (from assets folder)
model_path=Path("artifacts/model/model.keras")
model = load_model(model_path=model_path)

# Streamlit user interface
st.title("CNN Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction button
    if st.button('Classify Image'):
        class_label, confidence = predict_image(model, uploaded_file)
        st.write(f"Predicted Class: {class_label}")
        st.write(f"Confidence: {confidence:.2f}")
