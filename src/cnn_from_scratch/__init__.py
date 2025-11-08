"""
CNN From Scratch
================

Pure NumPy implementation of a small convolutional neural network along with
multilayer perceptron heads, activation functions, and training utilities.

The modules are organised so the project can be imported as a regular Python
package or executed via the command line entrypoints in `scripts/`.
"""

from .model import CNNClassifier
from .data import load_image_folder, train_test_split

__all__ = [
    "CNNClassifier",
    "load_image_folder",
    "train_test_split",
]

