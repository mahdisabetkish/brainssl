// Connectome explorer: the largest weights of the deployed classifier drawn
// between region centres, inside a translucent fsaverage5 pial surface.

import { Niivue } from "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/+esm";
import { getJSON } from "./app.js";
import { fmt, renderTable } from "./charts.js";

let nv = null;
let data = null;
let connectomeMesh = null;
const state = { atlas: null, k: 60, sign: "both" };

function selectEdges() {
  let edges = data.edges;
  if (state.sign === "pos") edges = edges.filter((e) => e[2] > 0);
  if (state.sign === "neg") edges = edges.filter((e) => e[2] < 0);
  return edges.slice(0, state.k);
}

// NiiVue draws an edge with radius |value| x edgeScale in mm, and the weights of
// a strongly penalised model are of order 1e-3, so edges get a display value:
// the sign of the weight, and a magnitude from 0.3 (weakest shown) to 1 (strongest).
const EDGE_FLOOR = 0.3;

function draw() {
  const shown = selectEdges();
  const n = data.names.length;
  const wmax = Math.max(...shown.map((e) => Math.abs(e[2])), 1e-12);
  const display = (w) => Math.sign(w) * (EDGE_FLOOR + (1 - EDGE_FLOOR) * Math.abs(w) / wmax);
  const matrix = new Array(n * n).fill(0);
  const strength = new Array(n).fill(0);
  for (const [i, j, w] of shown) {
    matrix[i * n + j] = display(w);
    matrix[j * n + i] = display(w);
    strength[i] += w / wmax;
    strength[j] += w / wmax;
  }
  const smax = Math.max(...strength.map(Math.abs), 1e-12);
  const touched = new Set(shown.flatMap((e) => [e[0], e[1]]));
  const nodes = {
    names: data.names, prefilled: [],
    X: data.xyz.map((p) => p[0]), Y: data.xyz.map((p) => p[1]), Z: data.xyz.map((p) => p[2]),
    Color: strength, Size: data.names.map((_, i) => (touched.has(i) ? 1 + 2.2 * Math.abs(strength[i]) / smax : 0)),
  };
  // loadConnectome() would clear every mesh, the brain surface included, so the
  // connectome is built as a mesh and swapped in by hand.
  if (connectomeMesh) nv.removeMesh(connectomeMesh);
  connectomeMesh = nv.loadConnectomeAsMesh({
    name: "weights", nodes, edges: matrix, showLegend: false,
    nodeColormap: "warm", nodeColormapNegative: "blue2cyan", nodeMinColor: 0, nodeMaxColor: smax, nodeScale: 3,
    edgeColormap: "warm", edgeColormapNegative: "blue2cyan", edgeMin: EDGE_FLOOR * 0.999, edgeMax: 1, edgeScale: 1.6,
  });
  nv.addMesh(connectomeMesh);
  nv.drawScene();
  document.getElementById("conn-k-label").textContent = `${shown.length} of the ${data.n_edges.toLocaleString()} connections in ${data.atlas}`;
  document.getElementById("conn-badge").textContent = `${data.atlas} · ${data.n_rois} regions`;
  renderTable(document.getElementById("conn-table"), [
    { key: "a", label: "Region" }, { key: "b", label: "Region" },
    { key: "w", label: "Weight ×10³", num: true, format: (v) => (v > 0 ? "+" : "") + fmt(v * 1000, 2) },
  ], shown.slice(0, 12).map(([i, j, w]) => ({ a: data.names[i], b: data.names[j], w })));
}

async function loadAtlas(atlas) {
  state.atlas = atlas;
  data = await getJSON(`data/connectome/${atlas}.json`);
  if (!data) return;
  draw();
}

export async function initConnectome() {
  const index = await getJSON("data/connectome/index.json");
  const sel = document.getElementById("conn-atlas");
  if (!index?.atlases?.length) {
    document.getElementById("conn-badge").textContent = "model not exported yet";
    return;
  }
  sel.innerHTML = index.atlases.map((a) => `<option value="${a.id}">${a.label}</option>`).join("");
  nv = new Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: false, isColorbar: false, isOrientCube: false,
    dragAndDropEnabled: false, scrollRequiresFocus: true, loadingText: "" });
  await nv.attachToCanvas(document.getElementById("conn-canvas"));
  await nv.loadMeshes([
    { url: "data/brain/pial_left.gii", rgba255: [205, 212, 224, 255], opacity: 0.1 },
    { url: "data/brain/pial_right.gii", rgba255: [205, 212, 224, 255], opacity: 0.1 },
  ]);
  nv.setRenderAzimuthElevation(90, 90);
  sel.addEventListener("change", (e) => loadAtlas(e.target.value));
  const k = document.getElementById("conn-k");
  k.addEventListener("input", () => { state.k = Number(k.value); draw(); });
  document.querySelectorAll("#conn-sign button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#conn-sign button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.sign = b.dataset.sign;
    draw();
  }));
  document.querySelectorAll("#conn-camera button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#conn-camera button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    nv.setRenderAzimuthElevation(Number(b.dataset.az), Number(b.dataset.el));
  }));
  await loadAtlas(index.atlases[0].id);
}
