
from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_iv.keras"
LABELS_PATH = ROOT / "models" / "mobilenetv2_labels.json"

IMAGE_SIZE = (224, 224)


def main():
    if len(sys.argv) != 2:
        print('Usage: python src\\predict_prototype.py "path\\to\\image.jpg"')
        raise SystemExit(2)

    image_path = Path(sys.argv[1]).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError(
            "Original MobileNetV2 model or labels file is missing."
        )

    with LABELS_PATH.open(encoding="utf-8") as file:
        labels = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(labels):
        raise ValueError("Model output count does not match saved labels.")

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    scores = model.predict(image, verbose=0)[0]
    order = np.argsort(scores)[::-1]
    predicted_label = labels[int(order[0])]

    print("\n=== IV FLUID CLASSIFICATION PROTOTYPE ===")
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {predicted_label}")
    print(f"Model score: {scores[order[0]]:.4f}")

    print("\nScores by class (not calibrated probabilities):")
    for index in order:
        print(f"  {labels[index]}: {scores[index]:.4f}")

    print("\nIMPORTANT:")
    print("- Research prototype only; not clinically validated.")
    print("- A model score is not a verified fluid percentage.")
    print("- Do not use this prediction to guide patient care.")
    print("- Confirm fluid status independently.")


if __name__ == "__main__":
    main()
