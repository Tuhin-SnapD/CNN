"""
Evaluation helpers for classification performance.
"""

from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions == targets))


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets, predictions, strict=True):
        matrix[int(true), int(pred)] += 1
    return matrix

