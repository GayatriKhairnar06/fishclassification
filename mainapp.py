<<<<<<< HEAD
# app.py - Streamlit Fish Classification App

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.markdown("Upload a fish image to predict its species.")

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "./VGG16_fish.h5"  # Replace with your saved model path
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Replace with your actual fish species names from training
CLASS_NAMES = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food horse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ---------------------------
# FILE UPLOADER
# ---------------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        # Preprocess
        processed_img = preprocess_image(image)

        # Prediction
        predictions = model.predict(processed_img)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        # Display results
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Show all probabilities
        st.subheader("Prediction Probabilities")
        for cls, prob in zip(CLASS_NAMES, predictions[0]):
            st.write(f"{cls}: {prob * 100:.2f}%")



=======
# app.py - Streamlit Fish Classification App

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.markdown("Upload a fish image to predict its species.")

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "./VGG16_fish.h5"  # Replace with your saved model path
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Replace with your actual fish species names from training
CLASS_NAMES = [
    "Species 1", "Species 2", "Species 3", "Species 4", "Species 5"
]

# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ---------------------------
# FILE UPLOADER
# ---------------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        # Preprocess
        processed_img = preprocess_image(image)

        # Prediction
        predictions = model.predict(processed_img)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        # Display results
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Show all probabilities
        st.subheader("Prediction Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{cls}: {predictions[0][i]*100:.2f}%")
>>>>>>> 7f91396 (Initial commit with model file using LFS)
