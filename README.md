CNN
================

Pure NumPy implementation of a tiny convolutional neural network (CNN) designed
for education and experimentation. The project shows every moving piece needed
to train an image classifier end to end—data loading, convolutions, activation
functions, multilayer perceptron heads, training loops, and evaluation metrics—
without relying on deep learning frameworks such as PyTorch or TensorFlow.

Features
--------

- **Fully NumPy-based pipeline** with explicit forward/backward passes for each layer.
- **Composable architecture** supporting multiple convolutional stages followed by an MLP head.
- **Dataset helpers** to load image folders into `(N, C, H, W)` tensors and split into train/validation sets.
- **Training CLI** (`scripts/train.py`) that saves learning curves for quick feedback.
- **Metrics & history tracking** including accuracy and a structured `TrainingHistory` object.

Project layout
--------------

- `src/cnn_from_scratch/`: Core library modules (`model.py`, `layers.py`, `activations.py`, `losses.py`, `metrics.py`, `data.py`, `mlp.py`).
- `scripts/`: Command line entry points such as `train.py`.
- `requirements.txt`: Minimal Python dependencies (NumPy, matplotlib, Pillow, tqdm).
- `runs/`: Default output directory for training artefacts (ignored via `.gitignore`).

Quick start
-----------

1. **Clone the repository**

   ```
   git clone https://github.com/Tuhin-SnapD/CNN.git
   cd CNN
   ```

2. **Create and activate a virtual environment**

   ```
   python -m venv .venv
   .venv\Scripts\activate        # PowerShell (Windows)
   # source .venv/bin/activate   # bash/zsh (macOS/Linux)
   ```

3. **Install dependencies**

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

Dataset preparation
-------------------

Place your images in class-specific subdirectories:

```
data/
├── cats/
│   ├── image_001.png
│   └── ...
└── dogs/
    ├── image_001.png
    └── ...
```

Supported formats include PNG and JPEG. Images are resized to the square
resolution given by `--image-size` (defaults to `28 28`). For grayscale data,
enable `--to-grayscale` when using the loading utilities programmatically.

Training from the command line
------------------------------

```
python scripts/train.py --data-dir data --epochs 5 --batch-size 64 --train-split 0.8
```

Arguments of note:

- `--image-size HEIGHT WIDTH` – resize all images before feeding them to the model.
- `--limit-per-class N` – cap the number of images per class (useful for quick experiments).
- `--no-plot` – skip saving the loss curve (`runs/loss_curve.png`).

Model API
---------

You can also build and train models directly in Python:

```python
import numpy as np
from cnn_from_scratch import CNNClassifier, load_image_folder, train_test_split

images, labels, class_names = load_image_folder("data", image_size=(28, 28))
x_train, x_val, y_train, y_val = train_test_split(images, labels, train_ratio=0.8, seed=42)

model = CNNClassifier(
    input_dim=28,
    input_channels=images.shape[1],
    conv_filter_sizes=(3, 3, 3),
    conv_num_filters=(8, 16, 32),
    conv_strides=(1, 1, 1),
    conv_activations=("relu", "relu", "relu"),
    mlp_hidden_layers=(len(class_names),),
    mlp_activations=("softmax",),
    learning_rate=0.001,
)

history = model.train(x_train, y_train, x_val, y_val, epochs=5, batch_size=64)
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation accuracy: {val_acc:.3f}")
```

Troubleshooting
---------------

- Ensure NumPy can allocate enough memory—training uses naïve Python loops for
  convolutions and can be slow for large images.
- If you see `FileNotFoundError` from `load_image_folder`, double-check the path
  and folder structure.
- Matplotlib may need a GUI backend on some systems; run with `--no-plot` in headless environments.

License
-------

MIT License. See `LICENSE` for the full text.

