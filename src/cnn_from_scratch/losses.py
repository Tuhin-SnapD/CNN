"""
Loss functions compatible with the NumPy training loop.
"""

from __future__ import annotations

import numpy as np


class Loss:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        diff = prediction - target
        return float(np.mean(np.sum(np.square(diff), axis=1)))

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        batch_size = prediction.shape[0]
        return (2.0 / batch_size) * (prediction - target)


class CrossEntropyLoss(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        batch_indices = np.arange(target.shape[0])
        clipped = np.clip(prediction, 1e-12, 1.0)
        log_likelihood = -np.log(clipped[batch_indices, target.astype(int)])
        return float(np.mean(log_likelihood))

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        batch_size, num_classes = prediction.shape
        clipped = np.clip(prediction, 1e-12, 1.0)
        grad = np.zeros_like(prediction)
        grad[np.arange(batch_size), target.astype(int)] = 1.0
        return -(grad - clipped) / batch_size

