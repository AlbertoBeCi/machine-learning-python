"""Image classification pipeline for Dogs vs Cats using EfficientNetB0."""

from pathlib import Path
import random
import shutil

from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = ("cat", "dog")
IMG_SIZE = (224, 224)


def prepare_dataset_splits(
    raw_train_dir: Path,
    processed_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """Split raw dog/cat images into train and test class subdirectories.

    Args:
        raw_train_dir: Directory containing raw ``cat.*.jpg`` and ``dog.*.jpg`` files.
        processed_dir: Target root directory where ``train/`` and ``test/`` will be created.
        test_size: Fraction of images per class assigned to the test split.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dictionary with counts for ``train`` and ``test`` splits.
    """
    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"

    existing_train = sum(len(list((train_dir / cls).glob("*.jpg"))) for cls in CLASSES if (train_dir / cls).exists())
    existing_test = sum(len(list((test_dir / cls).glob("*.jpg"))) for cls in CLASSES if (test_dir / cls).exists())
    if existing_train > 0 and existing_test > 0:
        return {"train": existing_train, "test": existing_test}

    rng = random.Random(seed)
    counts = {"train": 0, "test": 0}

    for cls in CLASSES:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

        files = sorted(raw_train_dir.glob(f"{cls}.*.jpg"))
        rng.shuffle(files)
        split_idx = int(len(files) * (1 - test_size))

        for split_name, subset in (("train", files[:split_idx]), ("test", files[split_idx:])):
            target_cls_dir = processed_dir / split_name / cls
            for src_path in subset:
                shutil.copy2(src_path, target_cls_dir / src_path.name)
            counts[split_name] += len(subset)

    return counts


def plot_first_nine_samples(
    raw_train_dir: Path,
    output_dir: Path,
    show: bool = False,
) -> tuple[Path, Path]:
    """Save 3x3 grids of the first nine dog and cat images.

    Args:
        raw_train_dir: Directory containing raw ``cat.*.jpg`` and ``dog.*.jpg`` files.
        output_dir: Directory where ``sample_dogs.png`` and ``sample_cats.png`` are saved.
        show: Whether to display figures interactively (useful in notebooks).

    Returns:
        Paths to the generated dog and cat sample figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for cls, title in (("dog", "Primeras 9 fotos de perros"), ("cat", "Primeras 9 fotos de gatos")):
        # Sort numerically by image index (e.g., dog.0.jpg .. dog.8.jpg)
        files = sorted(
            raw_train_dir.glob(f"{cls}.*.jpg"),
            key=lambda p: int(p.stem.split(".")[1]),
        )[:9]

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for ax, img_path in zip(axes.flat, files):
            with Image.open(img_path) as img:
                ax.imshow(img)
                ax.set_title(f"{img_path.name}\n{img.size[0]}x{img.size[1]}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        out_path = output_dir / f"sample_{cls}s.png"
        fig.savefig(out_path, dpi=150)
        saved_paths.append(out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_paths[0], saved_paths[1]


def create_data_generators(
    processed_dir: Path,
    target_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = 32,
):
    """Create Keras ImageDataGenerator flows for train and test directories.

    Args:
        processed_dir: Root directory containing ``train/`` and ``test/`` subfolders.
        target_size: Target spatial dimensions ``(height, width)`` for resizing.
        batch_size: Number of images per batch.

    Returns:
        Tuple ``(train_generator, test_generator)``.
    """
    # EfficientNetB0 includes an internal Rescaling(1./255) layer, so pixel values stay in [0, 255].
    trdata = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    tsdata = ImageDataGenerator()

    train_generator = trdata.flow_from_directory(
        directory=str(processed_dir / "train"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    test_generator = tsdata.flow_from_directory(
        directory=str(processed_dir / "test"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, test_generator


def build_efficientnet_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
    weights: str | None = None,
    freeze_base: bool = False,
) -> Sequential:
    """Build and compile the EfficientNetB0 binary classification model.

    Args:
        input_shape: Input tensor shape ``(height, width, channels)``.
        learning_rate: Adam optimizer learning rate.
        weights: Pretrained weights (``None`` or ``"imagenet"``).
        freeze_base: Whether to freeze the EfficientNetB0 backbone layers.

    Returns:
        Compiled Keras ``Sequential`` model.
    """
    base_model = EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    base_model.trainable = not freeze_base

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Sequential,
    train_generator,
    test_generator,
    checkpoint_path: Path,
    epochs: int = 10,
    patience: int = 3,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
):
    """Train the model with ModelCheckpoint and EarlyStopping callbacks.

    Args:
        model: Compiled Keras model.
        train_generator: Training directory iterator.
        test_generator: Validation/test directory iterator.
        checkpoint_path: File path to save the best model weights/architecture.
        epochs: Maximum number of training epochs.
        patience: Epochs with no validation improvement before stopping early.
        steps_per_epoch: Optional cap on batches per epoch.
        validation_steps: Optional cap on validation batches per epoch.

    Returns:
        Keras ``History`` object.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        verbose=1,
        restore_best_weights=True,
        mode="max",
    )

    return model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
    )


def evaluate_and_save_model(
    best_model_path: Path,
    final_model_path: Path,
    test_generator,
    steps: int | None = None,
) -> dict[str, object]:
    """Load the best checkpoint, evaluate on test data, and save the final model.

    Args:
        best_model_path: Path to the best checkpoint saved during training.
        final_model_path: Destination path for the final saved model.
        test_generator: Test directory iterator (must have ``shuffle=False``).
        steps: Optional number of batches to evaluate.

    Returns:
        Dictionary containing ``loss``, ``accuracy``, ``confusion_matrix``, and ``report``.
    """
    best_model = load_model(best_model_path)
    test_generator.reset()
    loss, accuracy = best_model.evaluate(test_generator, steps=steps, verbose=1)

    test_generator.reset()
    predictions = best_model.predict(test_generator, steps=steps, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes[: len(y_pred)]
    target_names = [k for k, _ in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    labels = list(range(len(target_names)))

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(final_model_path)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        ),
    }


def predict_pil_image(
    img: Image.Image,
    model,
    target_size: tuple[int, int] = IMG_SIZE,
) -> tuple[str, dict[str, float]]:
    """Predict whether a PIL image is a cat or a dog using a loaded model."""
    resized = img.convert("RGB").resize(target_size)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    probabilities = {cls: float(prob) for cls, prob in zip(CLASSES, probs)}
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class, probabilities


def predict_image(
    image_path: Path,
    model_path: Path,
    target_size: tuple[int, int] = IMG_SIZE,
) -> tuple[str, dict[str, float]]:
    """Predict whether a single image is a cat or a dog.

    Args:
        image_path: Path to the input image file.
        model_path: Path to the saved Keras ``.keras`` model.
        target_size: Target spatial dimensions ``(height, width)``.

    Returns:
        Tuple ``(predicted_label, probabilities)`` where probabilities maps
        each class name to its predicted confidence in ``[0, 1]``.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo guardado: {model_path}. Entrena primero con src/app.py."
        )

    model = load_model(model_path)
    with Image.open(image_path) as img:
        return predict_pil_image(img, model, target_size=target_size)
