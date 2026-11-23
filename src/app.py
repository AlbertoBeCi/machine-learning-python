"""Main entry point for the Dogs vs Cats EfficientNetB0 classification pipeline."""

import argparse
from pathlib import Path

from pipeline import (
    build_efficientnet_model,
    create_data_generators,
    evaluate_and_save_model,
    plot_first_nine_samples,
    prepare_dataset_splits,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_TRAIN_DIR = BASE_DIR / "data" / "raw" / "dogs-vs-cats" / "train"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    """Parse optional CLI arguments for training and evaluation."""
    parser = argparse.ArgumentParser(description="Dogs vs Cats EfficientNetB0 pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=3, help="EarlyStopping patience")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional batch cap per epoch")
    parser.add_argument("--validation-steps", type=int, default=None, help="Optional validation batch cap")
    parser.add_argument("--eval-steps", type=int, default=None, help="Optional test evaluation batch cap")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help='EfficientNetB0 weights: None (from scratch) or "imagenet" (transfer learning)',
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze EfficientNetB0 backbone layers (useful with --weights imagenet)",
    )
    return parser.parse_args()


def main() -> None:
    """Run data preparation, visualization, training, optimization, and saving."""
    args = parse_args()
    if not RAW_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_TRAIN_DIR}. "
            "Download and extract dogs-vs-cats.zip into data/raw first."
        )

    counts = prepare_dataset_splits(RAW_TRAIN_DIR, PROCESSED_DIR, test_size=0.2, seed=42)
    print(f"Dataset preparado -> train: {counts['train']} imágenes, test: {counts['test']} imágenes")

    dogs_fig, cats_fig = plot_first_nine_samples(RAW_TRAIN_DIR, INTERIM_DIR, show=False)
    print(f"Muestras guardadas en: {dogs_fig} y {cats_fig}")

    train_gen, test_gen = create_data_generators(PROCESSED_DIR, target_size=(224, 224), batch_size=32)

    model = build_efficientnet_model(
        input_shape=(224, 224, 3),
        learning_rate=1e-3,
        weights=args.weights,
        freeze_base=args.freeze_base,
    )
    model.summary()

    best_model_path = MODELS_DIR / "efficientnet_b0_dogs_vs_cats_best.keras"
    final_model_path = MODELS_DIR / "efficientnet_b0_dogs_vs_cats_final.keras"

    train_model(
        model=model,
        train_generator=train_gen,
        test_generator=test_gen,
        checkpoint_path=best_model_path,
        epochs=args.epochs,
        patience=args.patience,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )

    metrics = evaluate_and_save_model(
        best_model_path=best_model_path,
        final_model_path=final_model_path,
        test_generator=test_gen,
        steps=args.eval_steps,
    )
    print(f"\nEvaluación en test -> Loss: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    print("\nMatriz de confusión:\n", metrics["confusion_matrix"])
    print("\nReporte de clasificación:\n", metrics["report"])
    print(f"Modelo final guardado en: {final_model_path}")


if __name__ == "__main__":
    main()
