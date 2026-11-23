# 🐶🐱 Perros vs Gatos — Sistema de Clasificación de Imágenes

![Dogs vs Cats Image Classifier](assets/dogs_vs_cats_banner.png)

Clasificador binario de imágenes (**Gato** vs **Perro**) construido con **TensorFlow / Keras** y **EfficientNet-B0** sobre el dataset **Asirra (Dogs vs Cats)** (25.000 fotos `.jpg` etiquetadas). Incluye división automática del dataset, visualización de muestras, carga progresiva con `ImageDataGenerator`, callbacks `ModelCheckpoint` y `EarlyStopping`, soporte para *Transfer Learning* y una herramienta CLI para clasificar cualquier foto individual.

---

## ✨ Características y Resultados

- **Dataset**: 25.000 imágenes divididas en **80% entrenamiento** (`20.000`) / **20% prueba** (`5.000`), redimensionadas al vuelo a `224×224×3`.
- **Arquitectura**: `EfficientNetB0(include_top=False)` → `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dense(2, softmax)`.
- **Optimización**: `ModelCheckpoint` (guarda el mejor `val_accuracy`) + `EarlyStopping` (`patience=3`, restaura los mejores pesos).
- **Rendimiento en Test** (*Transfer Learning* con `--weights imagenet --freeze-base`, 3 épocas):
  - **Accuracy**: `98.92%` (`4.946 / 5.000` aciertos sobre todo el conjunto de test)
  - **Loss**: `0.0305`

---

## 📁 Estructura del Proyecto

- **`src/pipeline.py`** → Funciones modulares para dividir el dataset, graficar muestras, crear flujos `ImageDataGenerator`, construir/entrenar/evaluar `EfficientNetB0` e inferir sobre una imagen.
- **`src/app.py`** → Script principal CLI para entrenar, optimizar, evaluar y guardar el modelo.
- **`src/predict.py`** → Script CLI para clasificar una foto individual (`Gato` / `Perro`) con sus porcentajes de confianza.
- **`src/explore.ipynb`** → Notebook interactivo de Jupyter con todos los pasos del proyecto.
- **`src/utils.py`** → Funciones auxiliares de conexión a base de datos (`db_connect`).
- **`data/`** → `raw/` (zip original e imágenes extraídas), `interim/` (`sample_dogs.png`, `sample_cats.png`) y `processed/` (`train/` y `test/` por clase).
- **`models/`** → Checkpoints y modelo final `.keras`.

---

## 💻 Configuración en Local

### 1. Entorno Virtual y Dependencias (Python 3.11+)

```powershell
uv venv .venv --python 3.13 --seed
uv pip install -r requirements.txt ipykernel --python .venv\Scripts\python.exe
```

O usando `pip`:

```bash
pip install -r requirements.txt
```

### 2. Variables de Entorno (Base de datos opcional)

Crea un archivo `.env` en la raíz del proyecto si vas a usar PostgreSQL (por ejemplo con Docker):

```env
DATABASE_URL="postgresql+psycopg2://gitpod:postgres@localhost:5432/sample-db"
```

---

## 🚀 Uso

### Entrenar y Evaluar el Modelo

Entrenamiento completo desde cero (`weights=None`, 10 épocas):

```powershell
.venv\Scripts\python.exe src\app.py
```

Entrenamiento rápido con *Transfer Learning* (`98.92%` de accuracy en test en pocos minutos en CPU):

```powershell
.venv\Scripts\python.exe src\app.py --weights imagenet --freeze-base --epochs 3 --steps-per-epoch 100
```

Prueba rápida del flujo completo en CPU:

```powershell
.venv\Scripts\python.exe src\app.py --epochs 1 --steps-per-epoch 10 --validation-steps 10 --eval-steps 10
```

### Clasificar una Foto Individual

```powershell
.venv\Scripts\python.exe src\predict.py ruta\a\tu_foto.jpg
```

Ejemplo de salida:

```text
Predicción: Perro (99.98% de confianza)
  - Gato:  0.02%
  - Perro: 99.98%
```

---

## 🖼️ Visualizaciones de Muestra

Generadas automáticamente en `data/interim/`:

- [`data/interim/sample_dogs.png`](data/interim/sample_dogs.png)
- [`data/interim/sample_cats.png`](data/interim/sample_cats.png)
