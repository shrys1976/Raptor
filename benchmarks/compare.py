import json
import time
from pathlib import Path

import numpy as np

from raptor import Tensor, nn
from raptor.optim import Adam
from raptor.utils import (
    evaluate_classifier,
    load_mnist,
    save_comparison_curves,
    save_history_csv,
    save_history_json,
)


MODEL_DIMS = [(784, 128), (128, 64), (64, 10)]
RESULTS_PATH = Path("benchmarks/results/compare_results.json")
CURVES_DIR = Path("benchmarks/results/curves")


def make_raptor_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def generate_init_params(seed):
    rng = np.random.default_rng(seed)
    params = []

    for in_features, out_features in MODEL_DIMS:
        limit = np.sqrt(6.0 / in_features)
        weight = rng.uniform(-limit, limit, size=(out_features, in_features)).astype(np.float32)
        bias = np.zeros(out_features, dtype=np.float32)
        params.append({"weight": weight, "bias": bias})

    return params


def make_epoch_permutations(n_samples, epochs, seed):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_samples) for _ in range(epochs)]


def assign_raptor_params(model, init_params):
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]

    for layer, params in zip(linear_layers, init_params):
        layer.weight.data[...] = params["weight"]
        if layer.bias is not None:
            layer.bias.data[...] = params["bias"]


def train_raptor(
    X_train,
    y_train,
    X_test,
    y_test,
    init_params,
    epoch_permutations,
    epochs=5,
    batch_size=64,
    lr=1e-3,
    verbose=True,
):
    model = make_raptor_model()
    assign_raptor_params(model, init_params)

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

        perm = epoch_permutations[epoch]
        X_train_epoch = X_train[perm]
        y_train_epoch = y_train[perm]

        for i in range(0, len(X_train_epoch), batch_size):
            X_batch = X_train_epoch[i:i + batch_size]
            y_batch = y_train_epoch[i:i + batch_size]

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

        if verbose:
            print(
                f"[Raptor] epoch {epoch + 1}: "
                f"loss={train_loss:.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

    return model, history


def train_pytorch(
    X_train,
    y_train,
    X_test,
    y_test,
    init_params,
    epoch_permutations,
    epochs=5,
    batch_size=64,
    lr=1e-3,
    verbose=True,
):
    try:
        import torch
        import torch.nn as torch_nn
        import torch.optim as torch_optim
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install it first to run the benchmark."
        ) from exc

    class TorchMLP(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch_nn.Sequential(
                torch_nn.Linear(784, 128),
                torch_nn.ReLU(),
                torch_nn.Linear(128, 64),
                torch_nn.ReLU(),
                torch_nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = TorchMLP()

    linear_layers = [layer for layer in model.net if isinstance(layer, torch_nn.Linear)]
    with torch.no_grad():
        for layer, params in zip(linear_layers, init_params):
            layer.weight.copy_(torch.from_numpy(params["weight"]))
            layer.bias.copy_(torch.from_numpy(params["bias"]))

    criterion = torch_nn.CrossEntropyLoss()
    optimizer = torch_optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        start = time.perf_counter()
        perm = torch.tensor(epoch_permutations[epoch], dtype=torch.long)
        X_train_epoch = X_train_t[perm]
        y_train_epoch = y_train_t[perm]

        epoch_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for i in range(0, len(X_train_epoch), batch_size):
            X_batch = X_train_epoch[i:i + batch_size]
            y_batch = y_train_epoch[i:i + batch_size]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(X_batch)

        epoch_time = time.perf_counter() - start
        train_loss = epoch_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            preds = logits.argmax(dim=1)
            test_acc = (preds == y_test_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        if verbose:
            print(
                f"[PyTorch] epoch {epoch + 1}: "
                f"loss={train_loss:.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

    return model, history


def summarize_runs(histories):
    final_test_acc = np.array([run["test_acc"][-1] for run in histories], dtype=np.float64)
    final_train_acc = np.array([run["train_acc"][-1] for run in histories], dtype=np.float64)
    avg_epoch_time = np.array(
        [np.mean(run["epoch_time"]) for run in histories],
        dtype=np.float64,
    )

    return {
        "final_test_acc_mean": float(final_test_acc.mean()),
        "final_test_acc_std": float(final_test_acc.std()),
        "final_train_acc_mean": float(final_train_acc.mean()),
        "final_train_acc_std": float(final_train_acc.std()),
        "avg_epoch_time_mean": float(avg_epoch_time.mean()),
        "avg_epoch_time_std": float(avg_epoch_time.std()),
    }


def run_benchmark(
    seeds=(42, 43, 44),
    epochs=5,
    batch_size=64,
    lr=1e-3,
    data_dir="data/mnist",
):
    X_train, y_train, X_test, y_test = load_mnist(data_dir=data_dir)

    micro_runs = []
    torch_runs = []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n=== Run {run_idx}/{len(seeds)} | seed={seed} ===")
        init_params = generate_init_params(seed)
        epoch_permutations = make_epoch_permutations(len(X_train), epochs, seed + 10_000)

        _, raptor_history = train_raptor(
            X_train,
            y_train,
            X_test,
            y_test,
            init_params=init_params,
            epoch_permutations=epoch_permutations,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        micro_runs.append(raptor_history)

        _, torch_history = train_pytorch(
            X_train,
            y_train,
            X_test,
            y_test,
            init_params=init_params,
            epoch_permutations=epoch_permutations,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        torch_runs.append(torch_history)

    results = {
        "config": {
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "model_dims": MODEL_DIMS,
        },
        "raptor": {
            "runs": micro_runs,
            "summary": summarize_runs(micro_runs),
        },
        "pytorch": {
            "runs": torch_runs,
            "summary": summarize_runs(torch_runs),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    save_history_json(micro_runs[0], CURVES_DIR / "raptor_first_run.json")
    save_history_csv(micro_runs[0], CURVES_DIR / "raptor_first_run.csv")
    save_history_json(torch_runs[0], CURVES_DIR / "pytorch_first_run.json")
    save_history_csv(torch_runs[0], CURVES_DIR / "pytorch_first_run.csv")

    try:
        save_comparison_curves(
            {
                "Raptor": micro_runs[0],
                "PyTorch": torch_runs[0],
            },
            CURVES_DIR / "raptor_vs_pytorch.png",
            title="Raptor vs PyTorch",
        )
    except ModuleNotFoundError as exc:
        print(exc)

    return results


if __name__ == "__main__":
    results = run_benchmark()

    print("\nFinal Summary")
    print("Raptor:", results["raptor"]["summary"])
    print("PyTorch:   ", results["pytorch"]["summary"])
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved first-run histories under {CURVES_DIR}")
