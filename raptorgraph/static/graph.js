const statusBanner = document.getElementById("status-banner");
const nodeLayer = document.getElementById("node-layer");
const edgeLayer = document.getElementById("edge-layer");
const metaName = document.getElementById("meta-name");
const metaNodes = document.getElementById("meta-nodes");
const metaEdges = document.getElementById("meta-edges");

function setStatus(text) {
  statusBanner.textContent = text;
}

function formatPayload(payload) {
  if (!payload) {
    return "none";
  }

  if (payload.kind === "full") {
    return JSON.stringify(payload.values, null, 2);
  }

  return JSON.stringify(
    {
      preview: payload.preview,
      total_elements: payload.total_elements,
      min: payload.min,
      max: payload.max,
      mean: payload.mean,
    },
    null,
    2
  );
}

function buildDepthMap(nodes, edges) {
  const parentsByTarget = new Map();
  const nodeIds = new Set(nodes.map((node) => node.id));

  nodes.forEach((node) => parentsByTarget.set(node.id, []));
  edges.forEach((edge) => {
    if (nodeIds.has(edge.target)) {
      parentsByTarget.get(edge.target).push(edge.source);
    }
  });

  const depthCache = new Map();

  function depthOf(nodeId) {
    if (depthCache.has(nodeId)) {
      return depthCache.get(nodeId);
    }

    const parents = parentsByTarget.get(nodeId) || [];
    const depth = parents.length === 0 ? 0 : 1 + Math.max(...parents.map(depthOf));
    depthCache.set(nodeId, depth);
    return depth;
  }

  nodes.forEach((node) => depthOf(node.id));
  return depthCache;
}

function layoutNodes(nodes, edges) {
  const depthMap = buildDepthMap(nodes, edges);
  const columns = new Map();

  nodes.forEach((node) => {
    const depth = depthMap.get(node.id) || 0;
    if (!columns.has(depth)) {
      columns.set(depth, []);
    }
    columns.get(depth).push(node);
  });

  const positions = new Map();
  const columnGap = 230;
  const rowGap = 185;
  const startX = 22;
  const startY = 22;

  Array.from(columns.keys())
    .sort((a, b) => a - b)
    .forEach((depth) => {
      columns.get(depth).forEach((node, rowIndex) => {
        positions.set(node.id, {
          x: startX + depth * columnGap,
          y: startY + rowIndex * rowGap,
        });
      });
    });

  return positions;
}

function renderGraph(graph) {
  nodeLayer.innerHTML = "";
  edgeLayer.innerHTML = "";

  metaName.textContent = graph.name || "-";
  metaNodes.textContent = String(graph.node_count ?? graph.nodes.length);
  metaEdges.textContent = String(graph.edge_count ?? graph.edges.length);

  const positions = layoutNodes(graph.nodes, graph.edges);
  let maxX = 800;
  let maxY = 600;

  graph.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    maxX = Math.max(maxX, pos.x + 230);
    maxY = Math.max(maxY, pos.y + 190);

    const el = document.createElement("article");
    el.className = `graph-node ${node.requires_grad ? "requires-grad" : ""}`;
    el.style.left = `${pos.x}px`;
    el.style.top = `${pos.y}px`;

    el.innerHTML = `
      <div class="node-topline">
        <span class="node-op">${node.op}</span>
        <span class="node-shape">shape=${JSON.stringify(node.shape)}</span>
      </div>
      <div class="node-section">
        <h3>Data</h3>
        <pre class="node-pre">${formatPayload(node.data)}</pre>
      </div>
      <div class="node-section">
        <h3>Grad</h3>
        <pre class="node-pre">${formatPayload(node.grad)}</pre>
      </div>
      <div class="node-footer">${node.requires_grad ? "requires_grad=True" : "requires_grad=False"}</div>
    `;

    nodeLayer.appendChild(el);
  });

  edgeLayer.setAttribute("viewBox", `0 0 ${maxX} ${maxY}`);

  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) {
      return;
    }

    const x1 = source.x + 182;
    const y1 = source.y + 78;
    const x2 = target.x;
    const y2 = target.y + 78;
    const midX = (x1 + x2) / 2;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`
    );
    path.setAttribute("class", "edge-line");
    path.setAttribute("fill", "none");
    edgeLayer.appendChild(path);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(midX));
    label.setAttribute("y", String((y1 + y2) / 2 - 8));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "edge-label");
    label.textContent = edge.label || "";
    edgeLayer.appendChild(label);
  });

  nodeLayer.style.minWidth = `${maxX}px`;
  nodeLayer.style.minHeight = `${maxY}px`;
}

async function loadGraph() {
  setStatus("Loading graph...");
  try {
    const response = await fetch("/api/graph");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const graph = await response.json();
    renderGraph(graph);
    setStatus(`Loaded ${graph.name} with ${graph.node_count} nodes.`);
  } catch (error) {
    setStatus(`Could not load graph: ${error.message}`);
  }
}

loadGraph();
