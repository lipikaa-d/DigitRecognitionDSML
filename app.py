import streamlit as st
import numpy as np
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

st.title("🖐️ Handwritten Digit Recognition")
st.write("Upload a digit image and select a model to classify it!")

model_choice = st.sidebar.selectbox("Choose a model", ["SVM", "Random Forest", "CNN"])

uploaded_file = st.file_uploader("Upload a digit image (preferably 28x28 pixels)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Ensure correct size
    image_np = np.array(image)

    st.image(image, caption="Uploaded Digit", use_column_width=False, width=150)

    # Invert and flatten for traditional ML models
    img_inverted = 255 - image_np
    img_flat = img_inverted.flatten().reshape(1, -1)

    # Reshape for CNN model
    img_cnn = img_inverted.reshape(1, 28, 28, 1).astype("float32") / 255.0

    if st.button("Predict"):
        if model_choice == "SVM":
            with open("svm_model.pkl", "rb") as f:
                svm_model = pickle.load(f)
            prediction = svm_model.predict(img_flat)[0]
            st.success(f"🔢 Predicted Digit (SVM): {prediction}")

        elif model_choice == "Random Forest":
            with open("rf_model.pkl", "rb") as f:
                rf_model = pickle.load(f)
            prediction = rf_model.predict(img_flat)[0]
            st.success(f"🌲 Predicted Digit (Random Forest): {prediction}")

        elif model_choice == "CNN":
            cnn_model = load_model("cnn_model.h5")
            pred = cnn_model.predict(img_cnn)
            prediction = np.argmax(pred)
            confidence = np.max(pred)
            st.success(f"🧠 Predicted Digit (CNN): {prediction} with {confidence:.2%} confidence")
