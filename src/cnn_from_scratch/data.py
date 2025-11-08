"""
Data utilities for loading image datasets stored on disk.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def load_image_folder(
    root: str | Path,
    image_size: Tuple[int, int] = (28, 28),
    limit_per_class: int | None = None,
    to_grayscale: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load images arranged as `root/class_name/*.png`.

    Returns
    -------
    images : np.ndarray
        Array shaped `(n_samples, channels, height, width)` with values in [0, 1].
    labels : np.ndarray
        Label indices as integers.
    class_names : list[str]
        Ordered class names matching the label indices.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset folder {root_path} does not exist.")

    class_names = sorted(
        [p.name for p in root_path.iterdir() if p.is_dir()],
        key=lambda name: name.lower(),
    )
    if not class_names:
        raise ValueError(f"No class subfolders found in {root_path}.")

    images: list[np.ndarray] = []
    labels: list[int] = []
    channels = 1 if to_grayscale else 3

    for label_idx, class_name in enumerate(class_names):
        class_folder = root_path / class_name
        image_paths = sorted(class_folder.glob("*"))
        if limit_per_class is not None:
            image_paths = image_paths[:limit_per_class]
        for path in image_paths:
            with Image.open(path) as img:
                if to_grayscale:
                    img = img.convert("L")
                else:
                    img = img.convert("RGB")
                img = img.resize(image_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                if to_grayscale:
                    arr = arr[np.newaxis, :, :]
                else:
                    arr = arr.transpose(2, 0, 1)
                images.append(arr)
                labels.append(label_idx)

    if not images:
        raise ValueError(f"No images found in {root_path}.")
    images_np = np.stack(images, axis=0)
    labels_np = np.asarray(labels, dtype=np.int32)
    return images_np, labels_np, class_names


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    indices = list(range(len(x)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        x[train_idx],
        x[test_idx],
        y[train_idx],
        y[test_idx],
    )

