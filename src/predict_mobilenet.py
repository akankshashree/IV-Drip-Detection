
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_iv.keras"
LABELS_PATH = ROOT / "models" / "mobilenetv2_labels.json"

IMAGE_SIZE = (224, 224)


def load_model_and_labels():
    """Load the trained model and its matching class labels."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if len(labels) != model.output_shape[-1]:
        raise ValueError("Number of labels does not match model outputs.")

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")

    return model, labels


def preprocess_image(image):
    """Convert an image to RGB and apply training-time scaling."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = array / 127.5 - 1.0

    return np.expand_dims(array, axis=0)


def predict_image(model, labels, image):
    """Return the predicted category and raw model scores."""
    batch = preprocess_image(image)
    scores = model.predict(batch, verbose=0)[0]

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid numerical scores.")

    index = int(np.argmax(scores))

    return {
        "label": labels[index],
        "score": float(scores[index]),
        "scores": {
            label: float(score)
            for label, score in zip(labels, scores)
        },
    }
