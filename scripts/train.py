"""
Command line training entrypoint for the CNN from scratch project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cnn_from_scratch import CNNClassifier, load_image_folder, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset arranged as data_dir/class_name/image.*",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(28, 28))
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap on images loaded per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where training curves will be saved.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of dataset to use for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of learning curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    images, labels, class_names = load_image_folder(
        args.data_dir,
        image_size=tuple(args.image_size),
        limit_per_class=args.limit_per_class,
    )
    x_train, x_val, y_train, y_val = train_test_split(images, labels, train_ratio=args.train_split, seed=args.seed)

    model = CNNClassifier(
        input_dim=args.image_size[0],
        input_channels=images.shape[1],
        conv_filter_sizes=(3, 3, 3),
        conv_num_filters=(8, 16, 32),
        conv_strides=(1, 1, 1),
        conv_activations=("relu", "relu", "relu"),
        mlp_hidden_layers=(len(class_names),),
        mlp_activations=("softmax",),
        learning_rate=0.001,
    )

    history = model.train(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    val_preds = model.predict(x_val)
    accuracy = np.mean(val_preds == y_val)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(history.train_loss, label="train_loss")
        plt.plot(history.val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "loss_curve.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()

