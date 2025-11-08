"""
Core layer implementations for the NumPy-based CNN.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Array = np.ndarray


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float,
        regularisation: Literal["l1", "l2", "elastic", None] = None,
        reg_strength: tuple[float, float] | None = None,
    ) -> None:
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, (out_features, in_features))
        self.bias = np.zeros(out_features)
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.reg_strength = reg_strength or (0.0, 0.0)
        self._cache_inputs: Array | None = None
        self._cache_grad: Array | None = None

    def forward(self, x: Array) -> Array:
        self._cache_inputs = x
        return x @ self.weights.T + self.bias

    def backward(self, grad_output: Array) -> Array:
        if self._cache_inputs is None:
            raise RuntimeError("Linear.backward called before forward.")
        self._cache_grad = grad_output
        grad_input = grad_output @ self.weights
        return grad_input

    def step(self) -> None:
        if self._cache_inputs is None or self._cache_grad is None:
            raise RuntimeError("Linear.step called before forward/backward.")
        grad_w = self._cache_grad.T @ self._cache_inputs / self._cache_inputs.shape[0]
        grad_b = np.mean(self._cache_grad, axis=0)

        if self.regularisation == "l2":
            grad_w += self.reg_strength[1] * self.weights
        elif self.regularisation == "l1":
            grad_w += self.reg_strength[0] * np.sign(self.weights)
        elif self.regularisation == "elastic":
            grad_w += (
                self.reg_strength[0] * np.sign(self.weights)
                + self.reg_strength[1] * self.weights
            )

        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b


class Flatten:
    def __init__(self) -> None:
        self._from_shape: tuple[int, ...] | None = None

    def forward(self, x: Array) -> Array:
        self._from_shape = x.shape[1:]
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: Array) -> Array:
        if self._from_shape is None:
            raise RuntimeError("Flatten.backward called before forward.")
        return grad_output.reshape((grad_output.shape[0],) + self._from_shape)

    def step(self) -> None:
        """Flatten has no trainable parameters."""


class Convolution:
    def __init__(
        self,
        filter_size: int,
        num_filters: int,
        in_channels: int,
        input_dim: int,
        dilation: int = 1,
        learning_rate: float = 0.002,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.dilation = dilation
        self.learning_rate = learning_rate
        self.stride = stride
        self.padding = padding
        limit = np.sqrt(6.0 / (in_channels * filter_size * filter_size + num_filters))
        self.filters = np.random.uniform(
            -limit, limit, (num_filters, in_channels, filter_size, filter_size)
        )
        self._input_dim = input_dim
        self.feature_map_dim = int(
            ((input_dim + 2 * padding - self.effective_filter_size) / stride) + 1
        )
        self._cache_input: Array | None = None
        self._cache_grad: Array | None = None

    @property
    def effective_filter_size(self) -> int:
        return self.filter_size + (self.filter_size - 1) * (self.dilation - 1)

    def _pad(self, x: Array) -> Array:
        if self.padding == 0:
            return x
        return np.pad(
            x,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )

    def forward(self, x: Array) -> Array:
        padded = self._pad(x)
        batch_size = padded.shape[0]
        out_dim = self.feature_map_dim
        output = np.zeros((batch_size, self.num_filters, out_dim, out_dim))
        for b in range(batch_size):
            for f in range(self.num_filters):
                filter_weights = self.filters[f]
                for i in range(out_dim):
                    for j in range(out_dim):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = padded[
                            b,
                            :,
                            h_start : h_start + self.effective_filter_size : self.dilation,
                            w_start : w_start + self.effective_filter_size : self.dilation,
                        ]
                        output[b, f, i, j] = np.sum(region * filter_weights)
        self._cache_input = padded
        return output

    def backward(self, grad_output: Array) -> Array:
        if self._cache_input is None:
            raise RuntimeError("Convolution.backward called before forward.")

        batch_size = grad_output.shape[0]
        gradient_filters = np.zeros_like(self.filters)
        gradient_input = np.zeros_like(self._cache_input)

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(self.feature_map_dim):
                    for j in range(self.feature_map_dim):
                        grad_val = grad_output[b, f, i, j]
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = self._cache_input[
                            b,
                            :,
                            h_start : h_start + self.effective_filter_size : self.dilation,
                            w_start : w_start + self.effective_filter_size : self.dilation,
                        ]
                        gradient_filters[f] += grad_val * region
                        gradient_input[
                            b,
                            :,
                            h_start : h_start + self.effective_filter_size : self.dilation,
                            w_start : w_start + self.effective_filter_size : self.dilation,
                        ] += grad_val * self.filters[f]

        self._cache_grad = gradient_filters / batch_size
        if self.padding == 0:
            return gradient_input
        return gradient_input[
            :,
            :,
            self.padding : gradient_input.shape[2] - self.padding,
            self.padding : gradient_input.shape[3] - self.padding,
        ]

    def step(self) -> None:
        if self._cache_grad is None:
            raise RuntimeError("Convolution.step called before backward.")
        self.filters -= self.learning_rate * self._cache_grad

