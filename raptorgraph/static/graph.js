const statusBanner = document.getElementById("status-banner");
const nodeLayer = document.getElementById("node-layer");
const edgeLayer = document.getElementById("edge-layer");
const metaName = document.getElementById("meta-name");
const metaNodes = document.getElementById("meta-nodes");
const metaEdges = document.getElementById("meta-edges");
const graphPicker = document.getElementById("graph-picker");

function setStatus(text) {
  statusBanner.textContent = text;
}

function formatPayload(payload) {
  if (!payload) {
    return "none";
  }

  if (payload.kind === "full") {
    return `[${payload.values.join(", ")}]`;
  }

  return `preview=[${payload.preview.join(", ")}] n=${payload.total_elements} min=${payload.min.toFixed(3)} max=${payload.max.toFixed(3)} mean=${payload.mean.toFixed(3)}`;
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
  const columnGap = 210;
  const rowGap = 18;
  const startX = 22;
  const startY = 22;

  Array.from(columns.keys())
    .sort((a, b) => a - b)
    .forEach((depth) => {
      let currentY = startY;
      columns.get(depth).forEach((node) => {
        const el = nodeLayer.querySelector(`[data-node-id="${node.id}"]`);
        const height = el ? el.offsetHeight : 110;
        positions.set(node.id, {
          x: startX + depth * columnGap,
          y: currentY,
          height,
        });
        currentY += height + rowGap;
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

  let maxX = 800;
  let maxY = 600;

  graph.nodes.forEach((node) => {
    const el = document.createElement("article");
    el.className = `graph-node ${node.requires_grad ? "requires-grad" : ""}`;
    el.dataset.nodeId = node.id;

    el.innerHTML = `
      <div class="node-topline">
        <span class="node-kind">${node.op === "input" ? "input" : "tensor"}</span>
        <span class="node-shape">shape=${JSON.stringify(node.shape)}</span>
      </div>
      <div class="node-row">
        <span class="node-label">data</span>
        <span class="node-inline">${formatPayload(node.data)}</span>
      </div>
      <div class="node-row">
        <span class="node-label">grad</span>
        <span class="node-inline">${formatPayload(node.grad)}</span>
      </div>
      <div class="node-footer">${node.requires_grad ? "requires_grad=True" : "requires_grad=False"}</div>
    `;

    nodeLayer.appendChild(el);
  });

  const positions = layoutNodes(graph.nodes, graph.edges);

  graph.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    const el = nodeLayer.querySelector(`[data-node-id="${node.id}"]`);
    if (!pos || !el) {
      return;
    }
    el.style.left = `${pos.x}px`;
    el.style.top = `${pos.y}px`;
    maxX = Math.max(maxX, pos.x + el.offsetWidth + 24);
    maxY = Math.max(maxY, pos.y + pos.height + 24);
  });

  edgeLayer.setAttribute("viewBox", `0 0 ${maxX} ${maxY}`);
  edgeLayer.setAttribute("width", String(maxX));
  edgeLayer.setAttribute("height", String(maxY));
  edgeLayer.style.width = `${maxX}px`;
  edgeLayer.style.height = `${maxY}px`;

  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) {
      return;
    }

    const sourceEl = nodeLayer.querySelector(`[data-node-id="${edge.source}"]`);
    const targetEl = nodeLayer.querySelector(`[data-node-id="${edge.target}"]`);
    const sourceWidth = sourceEl ? sourceEl.offsetWidth : 160;
    const targetHeight = targetEl ? targetEl.offsetHeight : target.height;
    const x1 = source.x + sourceWidth;
    const y1 = source.y + source.height / 2;
    const x2 = target.x;
    const y2 = target.y + targetHeight / 2;
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
    label.textContent = edge.label === "input" ? "" : edge.label || "";
    edgeLayer.appendChild(label);
  });

  nodeLayer.style.minWidth = `${maxX}px`;
  nodeLayer.style.minHeight = `${maxY}px`;
  edgeLayer.style.minWidth = `${maxX}px`;
  edgeLayer.style.minHeight = `${maxY}px`;
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

async function loadDemo(name) {
  setStatus(`Loading ${name}...`);
  try {
    const response = await fetch(`/api/demo/${name}`, { method: "POST" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const graph = await response.json();
    renderGraph(graph);
    setStatus(`Loaded ${graph.name} with ${graph.node_count} nodes.`);
  } catch (error) {
    setStatus(`Could not load demo: ${error.message}`);
  }
}

async function initGraphPicker() {
  try {
    const response = await fetch("/api/demos");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    graphPicker.innerHTML = "";

    payload.demos.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      graphPicker.appendChild(option);
    });

    graphPicker.addEventListener("change", (event) => {
      loadDemo(event.target.value);
    });
  } catch (error) {
    setStatus(`Could not load demos: ${error.message}`);
  }
}

async function init() {
  await initGraphPicker();
  await loadGraph();
}

init();
