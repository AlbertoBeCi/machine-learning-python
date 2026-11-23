"""CLI script to classify a single image as cat or dog."""

import argparse
from pathlib import Path

from pipeline import predict_image

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "models" / "efficientnet_b0_dogs_vs_cats_final.keras"
LABELS_ES = {"cat": "Gato", "dog": "Perro"}


def main() -> None:
    """Parse image path and print the predicted class and probabilities."""
    parser = argparse.ArgumentParser(description="Clasifica una foto como perro o gato.")
    parser.add_argument("image", type=Path, help="Ruta a la imagen (.jpg, .png, etc.)")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Ruta al modelo .keras (por defecto: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    predicted_class, probabilities = predict_image(args.image, args.model)
    label_es = LABELS_ES.get(predicted_class, predicted_class)

    print(f"Imagen: {args.image}")
    print(f"Predicción: {label_es} ({probabilities[predicted_class]:.2%} de confianza)")
    print(f"  - Gato:  {probabilities['cat']:.2%}")
    print(f"  - Perro: {probabilities['dog']:.2%}")


if __name__ == "__main__":
    main()
