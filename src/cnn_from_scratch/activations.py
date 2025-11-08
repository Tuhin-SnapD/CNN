"""
Activation functions used across the project.

Every activation follows a minimal interface with `forward` and `backward`
methods. They operate on batches of inputs represented as NumPy arrays.
"""

from __future__ import annotations

import numpy as np


class Activation:
    """Base helper implementing the minimal protocol."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        activated = 1.0 / (1.0 + np.exp(-x))
        self._cache = activated
        return activated

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Sigmoid.backward called before forward.")
        derivative = self._cache * (1.0 - self._cache)
        return grad_output * derivative


class ReLU(Activation):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        activated = np.maximum(0.0, x)
        self._cache = x
        return activated

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("ReLU.backward called before forward.")
        mask = (self._cache > 0).astype(self._cache.dtype)
        return grad_output * mask


class Tanh(Activation):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        activated = np.tanh(x)
        self._cache = activated
        return activated

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Tanh.backward called before forward.")
        derivative = 1.0 - np.square(self._cache)
        return grad_output * derivative


class Softmax(Activation):
    def __init__(self) -> None:
        self._cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self._cache = probabilities
        return probabilities

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Softmax.backward called before forward.")
        grad_input = np.empty_like(grad_output)
        for i, (probs, grad) in enumerate(zip(self._cache, grad_output, strict=True)):
            jacobian = np.diag(probs) - np.outer(probs, probs)
            grad_input[i] = jacobian @ grad
        return grad_input

