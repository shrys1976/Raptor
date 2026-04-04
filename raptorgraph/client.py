import requests

from raptorgraph.tracer import GraphTracer


def register_graph(output_tensor, name, server_url="http://127.0.0.1:8000"):
    tracer = GraphTracer()
    graph = tracer.trace(output_tensor, name=name)

    response = requests.post(
        f"{server_url.rstrip('/')}/api/graphs/register",
        json={
            "name": name,
            "graph": graph,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
