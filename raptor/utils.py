import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .engine import Tensor


def batch_iterator(X, y, batch_size=32, shuffle=True):
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def accuracy_from_logits(logits, targets):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == targets)


def evaluate_classifier(model, X, y, batch_size=64):
    total_correct = 0
    total = 0

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        x = Tensor(X_batch, requires_grad=False)
        logits = model(x)

        preds = np.argmax(logits.data, axis=1)
        total_correct += np.sum(preds == y_batch)
        total += len(y_batch)

    return total_correct / total


def download_mnist(data_dir="data/mnist"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    for filename in files.values():
        path = data_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(base_url + filename, path)

    return {key: data_dir / filename for key, filename in files.items()}


def load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid MNIST image file magic number: {magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label file magic number: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir="data/mnist", normalize=True, flatten=True):
    paths = download_mnist(data_dir)

    X_train = load_mnist_images(paths["train_images"])
    y_train = load_mnist_labels(paths["train_labels"])
    X_test = load_mnist_images(paths["test_images"])
    y_test = load_mnist_labels(paths["test_labels"])

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    return X_train, y_train, X_test, y_test


def save_history_json(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    return path


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history.keys())
    lengths = {len(history[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError("all history series must have the same length")

    with path.open("w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(keys) + "\n")
        for idx in range(next(iter(lengths))):
            values = [str(history[key][idx]) for key in keys]
            f.write(f"{idx + 1}," + ",".join(values) + "\n")

    return path


def save_training_curves(history, path, title="Training Curves"):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save plotted training curves. "
            "You can still save the raw history with save_history_json/save_history_csv."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    for key in ("train_acc", "test_acc"):
        if key in history:
            axes[1].plot(epochs, history[key], label=key, linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_comparison_curves(histories, path, title="Framework Comparison"):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save plotted comparison curves."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for label, history in histories.items():
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{label} train_loss", linewidth=2)
        if "test_acc" in history:
            axes[1].plot(epochs, history["test_acc"], label=f"{label} test_acc", linewidth=2)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
