# 🐶🐱 Dogs vs Cats — Image Classification System

![Dogs vs Cats Image Classifier](assets/dogs_vs_cats_banner.png)

Binary image classifier (**Cat** vs **Dog**) built with **TensorFlow / Keras** and **EfficientNet-B0** on the **Asirra (Dogs vs Cats)** dataset (25,000 labeled `.jpg` photos). Includes automated dataset splitting, sample visualization, `ImageDataGenerator` streaming, `ModelCheckpoint` & `EarlyStopping` callbacks, Transfer Learning support, and a CLI tool to classify any single image.

---

## ✨ Key Features & Results

- **Dataset**: 25,000 images split **80% train** (`20,000`) / **20% test** (`5,000`), resized on the fly to `224×224×3`.
- **Architecture**: `EfficientNetB0(include_top=False)` → `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dense(2, softmax)`.
- **Optimization**: `ModelCheckpoint` (saves best `val_accuracy`) + `EarlyStopping` (`patience=3`, restores best weights).
- **Test Performance** (Transfer Learning with `--weights imagenet --freeze-base`, 3 epochs):
  - **Accuracy**: `98.92%` (`4,946 / 5,000` correct on full test set)
  - **Loss**: `0.0305`

---

## 📁 Project Structure

- **`src/pipeline.py`** → Modular functions for splitting data, plotting samples, creating `ImageDataGenerator` flows, building/training/evaluating `EfficientNetB0`, and single-image inference.
- **`src/app.py`** → Main CLI script for training, optimizing, evaluating, and saving the model.
- **`src/predict.py`** → CLI script to classify a single photo (`Gato` / `Perro`) with confidence scores.
- **`src/web.py`** & **`src/templates/index.html`** → Local Flask server and web UI to drag-and-drop, paste (`Ctrl+V`), or browse photos on your PC.
- **`src/explore.ipynb`** → Interactive Jupyter notebook following the project steps.
- **`src/utils.py`** → Auxiliary database connection utilities (`db_connect`).
- **`data/`** → `raw/` (original zip & extracted images), `interim/` (`sample_dogs.png`, `sample_cats.png`), and `processed/` (`train/` and `test/` class folders).
- **`models/`** → Saved `.keras` checkpoints and final model.

---

## 💻 Local Setup

### 1. Virtual Environment & Dependencies (Python 3.11+)

```powershell
uv venv .venv --python 3.13 --seed
uv pip install -r requirements.txt ipykernel --python .venv\Scripts\python.exe
```

Or with standard `pip`:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables (Optional DB)

Create a `.env` file in the root directory if using PostgreSQL (e.g. via Docker):

```env
DATABASE_URL="postgresql+psycopg2://gitpod:postgres@localhost:5432/sample-db"
```

---

## 🚀 Usage

### Train & Evaluate the Model

Full training from scratch (`weights=None`, 10 epochs):

```powershell
.venv\Scripts\python.exe src\app.py
```

Fast Transfer Learning training (`98.92%` test accuracy in a few minutes on CPU):

```powershell
.venv\Scripts\python.exe src\app.py --weights imagenet --freeze-base --epochs 3 --steps-per-epoch 100
```

Quick pipeline smoke test on CPU:

```powershell
.venv\Scripts\python.exe src\app.py --epochs 1 --steps-per-epoch 10 --validation-steps 10 --eval-steps 10
```

### Classify a Single Photo (CLI)

```powershell
.venv\Scripts\python.exe src\predict.py path\to\your_photo.jpg
```

### Launch the Local Web Interface

Start the local server (loads the model once and processes everything locally on your PC):

```powershell
.venv\Scripts\python.exe src\web.py
```

Then open **`http://127.0.0.1:5000`** in your browser to drag-and-drop, paste (`Ctrl+V`), or browse images from your computer.

Example output:

```text
Predicción: Perro (99.98% de confianza)
  - Gato:  0.02%
  - Perro: 99.98%
```

---

## 🖼️ Sample Data Visualizations

Generated automatically in `data/interim/`:

- [`data/interim/sample_dogs.png`](data/interim/sample_dogs.png)
- [`data/interim/sample_cats.png`](data/interim/sample_cats.png)
