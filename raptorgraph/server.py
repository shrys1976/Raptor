from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from raptor import Tensor, nn
from raptor.ops import relu, sigmoid, tanh
from raptor.utils import load_mnist
from raptorgraph.tracer import GraphTracer


STATIC_DIR = Path(__file__).parent / "static"
tracer = GraphTracer()
_graph_lock = Lock()
_current_graph = None
_saved_graphs = {}


class GraphRegistration(BaseModel):
    name: str
    graph: dict


def _set_graph(graph):
    global _current_graph
    with _graph_lock:
        _current_graph = graph


def get_current_graph():
    with _graph_lock:
        return _current_graph


def _save_graph(graph_id, graph):
    with _graph_lock:
        _saved_graphs[graph_id] = graph


def list_graphs():
    with _graph_lock:
        current_id = None if _current_graph is None else _current_graph.get("graph_id")
        saved_graph_ids = list(_saved_graphs.keys())

    graphs = []

    for demo_name in sorted(DEMO_BUILDERS.keys()):
        graph_id = f"demo:{demo_name}"
        graphs.append(
            {
                "id": graph_id,
                "label": demo_name,
                "type": "demo",
                "current": graph_id == current_id,
            }
        )

    for graph_id in saved_graph_ids:
        if graph_id.startswith("demo:"):
            continue
        if graph_id.startswith("custom:"):
            label = graph_id.split(":", 1)[1]
        else:
            label = graph_id

        graphs.append(
            {
                "id": graph_id,
                "label": label,
                "type": "custom",
                "current": graph_id == current_id,
            }
        )

    graphs.sort(key=lambda item: (item["type"] != "demo", item["label"]))
    return graphs


def set_current_tensor(output_tensor, name="custom", persist=True):
    graph = tracer.trace(output_tensor, name=name)
    graph_id = f"custom:{name}"
    graph["graph_id"] = graph_id
    graph["graph_type"] = "custom"
    if persist:
        _save_graph(graph_id, graph)
    _set_graph(graph)
    return graph


def register_graph_payload(name, graph_payload):
    graph = dict(graph_payload)
    graph_id = f"custom:{name}"
    graph["name"] = name
    graph["graph_id"] = graph_id
    graph["graph_type"] = "custom"
    _save_graph(graph_id, graph)
    _set_graph(graph)
    return graph


def _demo_arithmetic():
    a = Tensor(np.array([[1.0, -2.0], [3.0, 0.5]], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([[0.5, 4.0], [-1.0, 2.0]], dtype=np.float32), requires_grad=True)
    c = relu(a * b + a)
    out = c.mean()
    out.backward()
    return out


def _demo_mlp():
    np.random.seed(7)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=False)
    labels = np.array([0, 2], dtype=np.int64)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    return loss


def _demo_mnist_loss():
    np.random.seed(11)
    X_train, y_train, _, _ = load_mnist()

    model = nn.Sequential(
        nn.Linear(784, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    x = Tensor(X_train[:2], requires_grad=False)
    labels = y_train[:2]
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    return loss


DEMO_BUILDERS = {
    "arithmetic": _demo_arithmetic,
    "mlp": _demo_mlp,
    "mnist_loss": _demo_mnist_loss,
}


def load_demo(name):
    builder = DEMO_BUILDERS.get(name)
    if builder is None:
        raise KeyError(name)
    graph = tracer.trace(builder(), name=f"demo:{name}")
    graph_id = f"demo:{name}"
    graph["graph_id"] = graph_id
    graph["graph_type"] = "demo"
    _save_graph(graph_id, graph)
    _set_graph(graph)
    return graph


def activate_graph(graph_id):
    if graph_id.startswith("demo:"):
        demo_name = graph_id.split(":", 1)[1]
        return load_demo(demo_name)

    with _graph_lock:
        graph = _saved_graphs.get(graph_id)

    if graph is None:
        raise KeyError(graph_id)

    _set_graph(graph)
    return graph


app = FastAPI(title="RaptorGraph")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup():
    if get_current_graph() is None:
        load_demo("arithmetic")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/graph")
def api_graph():
    graph = get_current_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="No graph has been traced yet.")
    return graph


@app.get("/api/graphs")
def api_graphs():
    return {"graphs": list_graphs()}


@app.post("/api/demo/{name}")
def api_load_demo(name: str):
    try:
        graph = load_demo(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown demo: {name}") from exc
    return graph


@app.post("/api/graph/{graph_id:path}")
def api_activate_graph(graph_id: str):
    try:
        graph = activate_graph(graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown graph: {graph_id}") from exc
    return graph


@app.post("/api/graphs/register")
def api_register_graph(payload: GraphRegistration):
    return register_graph_payload(payload.name, payload.graph)
