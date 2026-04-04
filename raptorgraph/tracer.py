import numpy as np


class GraphTracer:
    def trace(self, output_tensor, name="graph"):
        nodes = []
        edges = []
        visited = set()

        def build(tensor):
            if tensor in visited:
                return
            visited.add(tensor)

            node_id = str(id(tensor))
            nodes.append(
                {
                    "id": node_id,
                    "op": tensor._op or "input",
                    "shape": list(tensor.data.shape),
                    "data": self._serialize_array(tensor.data),
                    "grad": self._serialize_array(tensor.grad),
                    "requires_grad": bool(tensor.requires_grad),
                }
            )

            for parent in tensor._prev:
                parent_id = str(id(parent))
                edges.append(
                    {
                        "source": parent_id,
                        "target": node_id,
                        "label": tensor._op or "input",
                    }
                )
                build(parent)

        build(output_tensor)
        return {
            "name": name,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def _serialize_array(self, array, max_elements=16):
        flat = array.astype(np.float32, copy=False).flatten()

        if flat.size <= max_elements:
            return {
                "kind": "full",
                "values": flat.tolist(),
            }

        return {
            "kind": "summary",
            "preview": flat[:max_elements].tolist(),
            "total_elements": int(flat.size),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
        }
