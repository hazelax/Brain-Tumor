import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')
class_names = ['Glioma', 'Meningioma', 'Pituitary']

# App title
st.title("🧠 Brain MRI Tumor Classifier")
st.write("Upload a brain MRI image to predict tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust size to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display results
    st.subheader("🧪 Prediction Results")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    # Show all class probabilities
    st.subheader("📊 Confidence Breakdown")
    for i, score in enumerate(predictions):
        st.write(f"{class_names[i]}: {score:.2f}")
