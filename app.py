import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="CNN Image Classifier", layout="centered")

st.title("🖼️ Image Classification Using CNN")
st.write("Upload an image to classify it using a trained CNN model.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_model.h5")

model = load_model()

# CIFAR-10 class labels
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image (PNG or JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((32, 32))
    img_array = np.array(img) / 255.0

    # Handle grayscale images
    if img_array.shape[-1] != 3:
        st.error("Please upload an RGB image.")
    else:
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.markdown("### 🔍 Prediction Result")
        st.write(f"**Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        
