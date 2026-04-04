import time
import numpy as np

from microtorch import Tensor, nn
from microtorch.optim import Adam
from microtorch.utils import load_mnist, batch_iterator, evaluate_classifier

import torch
import torch.nn as torch_nn
import torch.optim as torch_optim


np.random.seed(42)
X_train, y_train, X_test, y_test = load_mnist()


def make_microtorch_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def train_microtorch(epochs=5, batch_size=64, lr=1e-3):
    model = make_microtorch_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        start = time.perf_counter()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size=batch_size, shuffle=True):
            x = Tensor(X_batch, requires_grad=False)
            logits = model(x)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data) * len(X_batch)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == y_batch)
            total += len(X_batch)

        epoch_time = time.perf_counter() - start
        train_loss = epoch_loss / total
        train_acc = correct / total
        test_acc = evaluate_classifier(model, X_test, y_test, batch_size=batch_size)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"[MicroTorch] epoch {epoch+1}: "
            f"loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, "
            f"time={epoch_time:.2f}s"
        )

    return model, history
