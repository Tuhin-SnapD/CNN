"""
Fully-connected network used as the classifier head for the CNN.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .activations import ReLU, Sigmoid, Softmax, Tanh
from .layers import Linear


ActivationKey = str


def _make_activation(key: ActivationKey):
    match key.lower():
        case "relu" | "r":
            return ReLU()
        case "sigmoid" | "s":
            return Sigmoid()
        case "tanh" | "t":
            return Tanh()
        case "softmax":
            return Softmax()
        case _:
            raise ValueError(f"Unsupported activation key: {key!r}")


class MLP:
    def __init__(
        self,
        input_dim: int,
        layer_sizes: Sequence[int],
        activations: Sequence[ActivationKey],
        learning_rate: float,
        regularisation: str | None = None,
        reg_strength: tuple[float, float] | None = None,
    ) -> None:
        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one value.")
        if len(activations) != len(layer_sizes):
            raise ValueError("activations must match the number of layers.")

        self.linear_layers: list[Linear] = []
        self.activations = []
        prev_dim = input_dim
        for size, act in zip(layer_sizes, activations, strict=True):
            linear = Linear(
                prev_dim,
                size,
                learning_rate=learning_rate,
                regularisation=regularisation,
                reg_strength=reg_strength,
            )
            self.linear_layers.append(linear)
            self.activations.append(_make_activation(act))
            prev_dim = size

        self._last_activation = (
            isinstance(self.activations[-1], Softmax) if self.activations else False
        )
        self._cache_outputs: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_outputs = []
        out = x
        for linear, activation in zip(self.linear_layers, self.activations, strict=True):
            out = linear.forward(out)
            out = activation.forward(out)
            self._cache_outputs.append(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for linear, activation in zip(
            reversed(self.linear_layers), reversed(self.activations), strict=True
        ):
            grad = activation.backward(grad)
            grad = linear.backward(grad)
        return grad

    def step(self) -> None:
        for linear in self.linear_layers:
            linear.step()

