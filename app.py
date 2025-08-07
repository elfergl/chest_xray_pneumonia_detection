import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection from Chest X-ray")
st.write(
    "Upload a chest X-ray image and the model will predict whether pneumonia is present."
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/vgg16_finetuned_model.h5")
    return model


model = load_model()


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array


uploaded_file = st.file_uploader(
    "Choose a chest X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        input_array = preprocess_image(image_data)
        prediction = model.predict(input_array)[0][0]

        if prediction > 0.5:
            st.error(f"Prediction: Pneumonia detected (Confidence: {prediction:.2f})")
        else:
            st.success(f"Prediction: Normal (Confidence: {1 - prediction:.2f})")
