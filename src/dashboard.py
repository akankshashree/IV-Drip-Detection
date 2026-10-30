import streamlit as st
from PIL import Image, ImageOps

from level_logic import (
    ALIASES,
    CANONICAL,
    CHECK,
    CRITICAL,
    DISPLAY_NAMES,
    MIN_CONF,
    MIN_MARGIN,
    PAIR_MASS,
    WARNING,
    WATCH,
    decide,
)
from predict_mobilenet import load_model_and_labels, predict_image


st.set_page_config(
    page_title="IV Fluid Monitoring System",
    page_icon="💧",
    layout="wide",
)

KIND_TEXT = {
    "confident": "One level was clearly preferred by the model.",
    "between": (
        "Two neighbouring levels scored similarly, so the fluid is "
        "probably between them."
    ),
    "uncertain": "The model could not reach a reliable decision.",
}


def display_name(class_name):
    """Map a model class name to its dashboard label (Full, Enough, ...)."""
    canonical = ALIASES.get(class_name, class_name)
    if canonical in CANONICAL:
        return DISPLAY_NAMES[CANONICAL.index(canonical)]
    return class_name


@st.cache_resource
def get_model():
    return load_model_and_labels()


st.title("💧 IV Fluid Monitoring System")
st.caption(
    "Camera-based IV saline fill-level classification — research prototype"
)

st.warning(
    "Research prototype only. Predictions may be incorrect. "
    "Do not use this system for clinical decisions or to replace "
    "direct observation by trained healthcare staff."
)

with st.sidebar:
    st.header("Settings")
    with st.expander("Decision thresholds (for tuning)"):
        min_conf = st.slider(
            "Minimum top score to show a single level",
            0.30, 0.95, float(MIN_CONF), 0.05,
        )
        min_margin = st.slider(
            "Minimum lead over the runner-up",
            0.00, 0.60, float(MIN_MARGIN), 0.05,
        )
        pair_mass = st.slider(
            "Minimum combined score for an in-between state",
            0.50, 1.00, float(PAIR_MASS), 0.05,
        )
        st.info(
            "These only change how the model's raw scores are interpreted. "
            "They do not change the model. Scores are not calibrated "
            "probabilities, so tune these on photos of in-between levels."
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
        file_key = f"{uploaded_file.name}-{uploaded_file.size}"

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
                # Keep the result so moving a slider does not erase it.
                st.session_state["analysis"] = {
                    "key": file_key,
                    "result": result,
                    "class_names": list(labels),
                }
            except Exception as error:
                st.session_state.pop("analysis", None)
                st.error(f"Prediction failed: {error}")

        analysis = st.session_state.get("analysis")

        if analysis and analysis["key"] == file_key:
            try:
                result = analysis["result"]
                class_names = analysis["class_names"]
                probs = [result["scores"][name] for name in class_names]

                state = decide(
                    probs,
                    class_names,
                    min_conf=min_conf,
                    min_margin=min_margin,
                    pair_mass=pair_mass,
                )

                st.divider()
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated level", state.label)
                with col2:
                    st.metric("Alert", state.alert_name)

                if state.alert == CRITICAL:
                    st.error("DEMONSTRATION ALERT: Empty bottle predicted")
                elif state.alert == WARNING:
                    st.warning(
                        "DEMONSTRATION WARNING: Low fluid level predicted"
                    )
                elif state.alert == WATCH:
                    st.info(
                        "Fluid is around half a bottle. Keep monitoring."
                    )
                elif state.alert == CHECK:
                    st.warning(
                        "The model could not decide. Check the bottle "
                        "directly."
                    )
                else:
                    st.success("Level appears adequate")

                st.caption(KIND_TEXT[state.kind])

                with st.expander("View raw scores for all levels"):
                    for name in class_names:
                        st.write(
                            f"{display_name(name)} ({name}): "
                            f"{result['scores'][name]:.4f}"
                        )
                    st.caption(
                        "Raw model scores, not calibrated probabilities "
                        "and not a guarantee of correctness."
                    )

                st.caption(
                    "The levels are nominal categories, not a direct "
                    "measurement of remaining fluid volume. The model has "
                    "not been validated for clinical use or independent "
                    "bottles."
                )

            except Exception as error:
                st.error(f"Could not interpret the prediction: {error}")

    except Exception as error:
        st.error(f"Could not open the uploaded image: {error}")

st.divider()
st.caption(
    "IV Fluid Monitoring Prototype | Not clinically validated | "
    "Always verify fluid levels independently."
)