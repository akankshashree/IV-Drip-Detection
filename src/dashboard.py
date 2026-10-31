
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hog_linear_svc.joblib"

LEVELS = {
    "sal_data_100": 100,
    "sal_data_80": 80,
    "sal_data_50": 50,
    "sal_data_empty": 0,
}

LEVEL_LABELS = {
    "sal_data_100": "100% filled",
    "sal_data_80": "80% filled",
    "sal_data_50": "50% filled",
    "sal_data_empty": "Empty",
}

st.set_page_config(
    page_title="IV Fluid Monitoring System",
    page_icon="💧",
    layout="wide",
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run python src\\train_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    return saved["model"], tuple(saved["image_size"])


# ---------------- IMAGE PREPROCESSING ----------------

def extract_features(image_rgb, image_size):
    """
    Match the training pipeline:
    resize -> BGR-to-grayscale equivalent -> HOG features.

    Uploaded images are RGB, so use RGB-to-grayscale.
    During training, cv2.imread produced BGR images, so
    training used BGR-to-grayscale.
    """
    resized = cv2.resize(image_rgb, image_size)

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    hog = cv2.HOGDescriptor(
        (96, 96),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )

    features = hog.compute(gray).flatten().reshape(1, -1)

    return features


def predict_level(image_rgb, model, image_size):
    features = extract_features(image_rgb, image_size)
    predicted_class = model.predict(features)[0]

    return predicted_class, LEVELS[predicted_class]


# ---------------- DASHBOARD UI ----------------

st.title("💧 IV Fluid Monitoring System")
st.caption(
    "Camera-based IV fluid category classification — research prototype"
)

st.warning(
    "Research prototype only. Predictions may be incorrect. "
    "Do not use this system to make clinical decisions or replace "
    "direct observation by trained healthcare staff."
)

with st.sidebar:
    st.header("Settings")

    threshold = st.slider(
        "Low-level warning threshold (%)",
        min_value=0,
        max_value=80,
        value=50,
        step=10,
    )

    st.info(
        "The threshold controls the demonstration warning only. "
        "It does not change the model's prediction."
    )

st.subheader("Upload an IV bottle image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)

        left, right = st.columns([1, 1])

        with left:
            st.image(
                pil_image,
                caption="Uploaded IV bottle image",
                use_container_width=True,
            )

        with right:
            st.subheader("Image information")
            st.write(f"Width: {pil_image.width} pixels")
            st.write(f"Height: {pil_image.height} pixels")
            st.write(f"Format: {pil_image.format or 'Converted to RGB'}")

        if st.button("Analyze Fluid Level", type="primary"):
            try:
                model, image_size = load_model()

                with st.spinner("Analyzing image..."):
                    predicted_class, level = predict_level(
                        image_rgb,
                        model,
                        image_size,
                    )

                st.divider()
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Estimated category",
                        LEVEL_LABELS[predicted_class],
                    )

                with col2:
                    st.metric(
                        "Nominal category value",
                        f"{level}%",
                    )

                st.write("Predicted class:", predicted_class)

                if level == 0:
                    st.error("DEMONSTRATION ALERT: Empty category predicted")
                elif level <= threshold:
                    st.warning("DEMONSTRATION WARNING: Low category predicted")
                else:
                    st.success("Above demonstration threshold")

                st.progress(level / 100)

                st.caption(
                    "The percentage is a category label, not a direct "
                    "measurement of the remaining fluid volume. "
                    "The classifier may misclassify unfamiliar images."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    except Exception as error:
        st.error(f"Could not open the uploaded image: {error}")

else:
    st.info("Upload an image to begin.")

st.divider()
st.caption(
    "IV Fluid Monitoring Prototype | Not clinically validated | "
    "Always verify the fluid level independently."
)
