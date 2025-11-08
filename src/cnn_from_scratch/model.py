"""
High level CNN classifier composed of convolutional and dense layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm

from .activations import ReLU, Sigmoid, Softmax, Tanh
from .layers import Convolution, Flatten
from .losses import CrossEntropyLoss, Loss
from .metrics import accuracy
from .mlp import MLP


ActivationKey = str


def _make_activation(key: ActivationKey):
    match key.lower():
        case "relu" | "r":
            return ReLU()
        case "sigmoid" | "s":
            return Sigmoid()
        case "tanh" | "t":
            return Tanh()
        case _:
            raise ValueError(f"Unsupported activation key for CNN: {key!r}")


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    train_accuracy: list[float]
    val_accuracy: list[float]


class CNNClassifier:
    def __init__(
        self,
        input_dim: int,
        input_channels: int,
        conv_filter_sizes: Sequence[int],
        conv_num_filters: Sequence[int],
        conv_strides: Sequence[int],
        conv_activations: Sequence[ActivationKey],
        mlp_hidden_layers: Sequence[int],
        mlp_activations: Sequence[ActivationKey],
        learning_rate: float = 0.001,
        loss: Loss | None = None,
    ) -> None:
        if not (
            len(conv_filter_sizes)
            == len(conv_num_filters)
            == len(conv_strides)
            == len(conv_activations)
        ):
            raise ValueError("Convolutional layer configuration lengths do not match.")

        self.conv_layers: list[object] = []
        current_channels = input_channels
        current_dim = input_dim
        for filter_size, num_filters, stride, activation in zip(
            conv_filter_sizes, conv_num_filters, conv_strides, conv_activations, strict=True
        ):
            conv = Convolution(
                filter_size=filter_size,
                num_filters=num_filters,
                in_channels=current_channels,
                input_dim=current_dim,
                stride=stride,
                learning_rate=learning_rate,
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(_make_activation(activation))
            current_channels = num_filters
            current_dim = conv.feature_map_dim

        self.conv_layers.append(Flatten())
        mlp_input_dim = current_dim * current_dim * current_channels

        self.mlp = MLP(
            input_dim=mlp_input_dim,
            layer_sizes=mlp_hidden_layers,
            activations=mlp_activations,
            learning_rate=learning_rate,
        )
        self.loss_fn = loss or CrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.conv_layers:
            out = layer.forward(out)
        return self.mlp.forward(out)

    def backward(self, grad_output: np.ndarray) -> None:
        grad = self.mlp.backward(grad_output)
        for layer in reversed(self.conv_layers):
            grad = layer.backward(grad)

    def step(self) -> None:
        for layer in self.conv_layers:
            if hasattr(layer, "step"):
                layer.step()
        self.mlp.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        probs = self.forward(x)
        preds = np.argmax(probs, axis=1)
        return self.loss_fn.forward(probs, y), accuracy(preds, y)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> TrainingHistory:
        history = TrainingHistory([], [], [], [])
        num_batches = int(np.ceil(len(x_train) / batch_size))

        for epoch in range(epochs):
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            iterator: Iterable = range(num_batches)
            if verbose:
                iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = min(len(x_train), start + batch_size)
                inputs = x_train_shuffled[start:end]
                targets = y_train_shuffled[start:end]
                outputs = self.forward(inputs)
                loss_value = self.loss_fn.forward(outputs, targets)
                grad_output = self.loss_fn.backward(outputs, targets)
                self.backward(grad_output)
                self.step()

                if verbose and isinstance(iterator, tqdm):
                    iterator.set_postfix({"loss": f"{loss_value:.4f}"})

            train_loss, train_acc = self.evaluate(x_train, y_train)
            val_loss, val_acc = self.evaluate(x_val, y_val)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

        return history

