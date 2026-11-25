"""Local Flask web app for classifying dog and cat images."""

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from PIL import Image, UnidentifiedImageError

from pipeline import predict_pil_image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_dogs_vs_cats_final.keras"
LABELS_ES = {"cat": "Gato", "dog": "Perro"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.config["TEMPLATES_AUTO_RELOAD"] = True

_MODEL = None


def get_model():
    """Load and cache the trained Keras model once per server process."""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODEL_PATH}. Entrena primero con src/app.py."
            )
        _MODEL = load_model(MODEL_PATH)
    return _MODEL


@app.get("/")
def index():
    """Render the local image classification page."""
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    """Classify an uploaded image file as cat or dog."""
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "El archivo seleccionado está vacío."}), 400

    try:
        with Image.open(file.stream) as img:
            predicted_class, probabilities = predict_pil_image(img, get_model())
    except UnidentifiedImageError:
        return jsonify({"error": "El archivo no es una imagen válida (.jpg, .png, .webp)."}), 400
    except Exception as exc:
        return jsonify({"error": f"No se pudo procesar la imagen: {exc}"}), 500

    return jsonify(
        {
            "label": predicted_class,
            "label_es": LABELS_ES.get(predicted_class, predicted_class),
            "confidence": probabilities[predicted_class],
            "probabilities": probabilities,
        }
    )


if __name__ == "__main__":
    get_model()
    print("Servidor local listo en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
