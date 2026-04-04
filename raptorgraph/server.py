from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from raptor import Tensor, nn
from raptor.ops import relu, sigmoid, tanh
from raptorgraph.tracer import GraphTracer


STATIC_DIR = Path(__file__).parent / "static"
tracer = GraphTracer()
_graph_lock = Lock()
_current_graph = None


def _set_graph(graph):
    global _current_graph
    with _graph_lock:
        _current_graph = graph


def get_current_graph():
    with _graph_lock:
        return _current_graph


def set_current_tensor(output_tensor, name="custom"):
    graph = tracer.trace(output_tensor, name=name)
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


DEMO_BUILDERS = {
    "arithmetic": _demo_arithmetic,
    "mlp": _demo_mlp,
}


def load_demo(name):
    builder = DEMO_BUILDERS.get(name)
    if builder is None:
        raise KeyError(name)
    return set_current_tensor(builder(), name=f"demo:{name}")


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


@app.get("/api/demos")
def api_demos():
    return {"demos": sorted(DEMO_BUILDERS.keys())}


@app.post("/api/demo/{name}")
def api_load_demo(name: str):
    try:
        graph = load_demo(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown demo: {name}") from exc
    return graph
