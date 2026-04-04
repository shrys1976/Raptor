import numpy as np
from .engine import Tensor
import gzip
import struct
import urllib.request
from pathlib import Path


#batch iterator
def batch_iterator(X,y,batch_size=32, shuffle=True):
    n = len(X)

    indices = np.arrange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0,n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_size],y[batch_idx]   


def accuracy_from_logits(logits,targets):
    preds = np.argmax(logits,axis = 1)
    return np.mean(preds ==targets)        




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


#mnist dataset helpers

# dataset downlaoder

def download_mnist(data_dir = "data/mnist"):
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


# image parser fr idx format

def load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images



#label idx formant parser
def load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

#main loader
   
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

