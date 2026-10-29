
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageOps

from predict_mobilenet import load_model_and_labels, predict_image


st.set_page_config(
    page_title="IV Fluid Monitoring System",
    page_icon="💧",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent

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


@st.cache_resource
def get_model():
    return load_model_and_labels()


st.title("💧 IV Fluid Monitoring System")
st.caption(
    "Camera-based IV saline fill-category classification — research prototype"
)

st.warning(
    "Research prototype only. Predictions may be incorrect. "
    "Do not use this system for clinical decisions or to replace "
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
        "This threshold controls a demonstration warning only. "
        "It does not change the model prediction."
    )

st.subheader("Upload an IV bottle image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded IV bottle image",
                use_container_width=True,
            )

        with right:
            st.subheader("Image information")
            st.write(f"Width: {image.width} pixels")
            st.write(f"Height: {image.height} pixels")
            st.write("Format: RGB")

        if st.button("Analyze Fluid Level", type="primary"):
            try:
                with st.spinner("Loading model and analyzing image..."):
                    model, labels = get_model()
                    result = predict_image(model, labels, image)

                predicted_class = result["label"]
                level = LEVELS.get(predicted_class)

                if level is None:
                    raise ValueError(
                        f"Unexpected predicted class: {predicted_class}"
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
                    st.metric("Nominal category value", f"{level}%")

                st.write("Predicted class:", predicted_class)

                st.caption(
                    f"Raw model score: {result['score']:.1%}. "
                    "This score is not a calibrated probability or "
                    "a guarantee of prediction correctness."
                )

                if level == 0:
                    st.error(
                        "DEMONSTRATION ALERT: Empty category predicted"
                    )
                elif level <= threshold:
                    st.warning(
                        "DEMONSTRATION WARNING: Low category predicted"
                    )
                else:
                    st.success("Above demonstration threshold")

                st.progress(level / 100)

                with st.expander("View scores for all categories"):
                    for label in labels:
                        st.write(
                            f"{LEVEL_LABELS.get(label, label)}: "
                            f"{result['scores'][label]:.4f}"
                        )

                st.caption(
                    "The displayed percentage is a nominal dataset "
                    "category, not a direct measurement of remaining "
                    "fluid volume. The model has not been validated "
                    "for clinical use or independent bottles."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    except Exception as error:
        st.error(f"Could not open the uploaded image: {error}")

st.divider()
st.caption(
    "IV Fluid Monitoring Prototype | Not clinically validated | "
    "Always verify fluid levels independently."
)

