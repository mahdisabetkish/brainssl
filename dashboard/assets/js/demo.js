// Live demos: the connectome classifier in a Web Worker, and the segmentation
// network through ONNX Runtime Web.

import { getJSON } from "./app.js";
import { css, d3, fmt, hideTip, showTip } from "./charts.js";

const HF_MODEL = "https://huggingface.co/mahdisabetkish/brainssl-tumour-segmentation/resolve/main/segmentation.onnx";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

// ---------------------------------------------------------------- fMRI

let fmriIndex = null;
const buffers = new Map();

async function floats(path) {
  if (!buffers.has(path)) buffers.set(path, fetch(path).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)));
  return buffers.get(path);
}

function windowChart(el, res) {
  el.innerHTML = "";
  const w = Math.max(300, el.clientWidth || 500), h = 150;
  const m = { top: 12, right: 16, bottom: 34, left: 40 };
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
  const iw = w - m.left - m.right, ih = h - m.top - m.bottom;
  const items = [{ label: "full", p: res.full }, ...res.perWindow.map((p, i) => ({ label: `w${i + 1}`, p }))];
  const x = d3.scaleBand().domain(items.map((d) => d.label)).range([0, iw]).padding(0.45);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  g.append("g").attr("class", "grid").selectAll("line").data([0, 0.25, 0.5, 0.75, 1]).join("line")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6)).call((a) => a.select(".domain").remove());
  g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(0.5)).attr("y2", y(0.5)).attr("stroke", css("--axis"));
  const bw = Math.min(24, x.bandwidth());
  g.append("g").selectAll("path").data(items).join("path")
    .attr("d", (d) => {
      const x0 = x(d.label) + (x.bandwidth() - bw) / 2, y0 = y(d.p), r = Math.min(4, (ih - y0) / 2);
      return `M${x0},${ih}V${y0 + r}Q${x0},${y0} ${x0 + r},${y0}H${x0 + bw - r}Q${x0 + bw},${y0} ${x0 + bw},${y0 + r}V${ih}Z`;
    })
    .attr("fill", (d) => (d.label === "full" ? css("--series-1") : css("--series-6")))
    .on("mousemove", (event, d) => showTip(event, d.label === "full" ? "full scan" : `window ${d.label.slice(1)}`, [["P(autism)", fmt(d.p)]]))
    .on("mouseleave", hideTip);
}

async function runFmri() {
  const subject = fmriIndex.subjects.find((s) => s.id === document.getElementById("demo-subject").value);
  const atlas = document.getElementById("demo-atlas").value;
  const model = fmriIndex.model.atlases[atlas];
  const out = document.getElementById("demo-out");
  const spec = subject.atlases[atlas];
  out.innerHTML = `<div class="progress"><i id="demo-progress"></i></div>`;
  const [ts, params] = await Promise.all([floats(`data/demo/fmri/${spec.file}`), floats(`data/demo/fmri/${model.file}`)]);
  const worker = new Worker(new URL("./demo-worker.js", import.meta.url), { type: "module" });
  const started = performance.now();
  worker.onmessage = (event) => {
    if (event.data.progress != null) {
      const bar = document.getElementById("demo-progress");
      if (bar) bar.style.width = `${Math.round(100 * event.data.progress)}%`;
      return;
    }
    worker.terminate();
    if (event.data.error) { out.textContent = event.data.error; return; }
    const res = event.data.result;
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const pred = res.probability >= 0.5 ? "autism" : "control";
    const truth = subject.label === 1 ? "autism" : "control";
    out.innerHTML = `<div class="grid g-3" style="gap:12px">
        <div class="stat"><div class="label">P(autism)</div><div class="value">${fmt(res.probability, 2)}</div></div>
        <div class="stat"><div class="label">Prediction</div><div class="value" style="font-size:24px;margin-top:12px">${pred}</div></div>
        <div class="stat"><div class="label">Recorded diagnosis</div><div class="value" style="font-size:24px;margin-top:12px">${truth}</div></div>
      </div>`;
    windowChart(document.getElementById("demo-chart"), res);
    document.getElementById("demo-note").textContent =
      `${spec.T} volumes x ${spec.R} regions; full scan plus ${res.perWindow.length} windows of ${res.width} volumes; ` +
      `computed in your browser in ${seconds} s. The Python model gives ${fmt(spec.python_probability, 3)} for this scan.`;
  };
  worker.postMessage({ ts, T: spec.T, R: spec.R, params, bias: model.bias, windows: fmriIndex.model.windows });
}

// ---------------------------------------------------------------- MRI

let segViewer = null;
let segWorker = null;

// One request at a time to the worker that holds the network.
function ask(worker, message, transfer = []) {
  return new Promise((resolve, reject) => {
    worker.onmessage = (event) => (event.data.type === "error" ? reject(new Error(event.data.message)) : resolve(event.data));
    worker.onerror = (event) => reject(new Error(event.message || "the model worker stopped"));
    worker.postMessage(message, transfer);
  });
}

function stopSegmentation(message) {
  if (segWorker) { segWorker.terminate(); segWorker = null; }
  const button = document.getElementById("seg-run");
  button.innerHTML = '<svg><use href="#i-play"/></svg>Run segmentation';
  if (message) document.getElementById("seg-out").textContent = message;
}

function hasWebGL2() {
  try { return !!document.createElement("canvas").getContext("webgl2"); } catch { return false; }
}

function starts(size, win, overlap = 0.5) {
  if (size <= win) return [0];
  const step = Math.max(1, Math.floor(win * (1 - overlap)));
  const out = [];
  for (let s = 0; s <= size - win; s += step) out.push(s);
  if (out[out.length - 1] !== size - win) out.push(size - win);
  return out;
}

function gaussian(shape, scale = 0.125) {
  const axis = (n) => Array.from({ length: n }, (_, i) => Math.exp(-0.5 * ((i - (n - 1) / 2) / (scale * n)) ** 2));
  const [a, b, c] = shape.map(axis);
  const w = new Float32Array(shape[0] * shape[1] * shape[2]);
  let max = 0;
  for (let i = 0; i < shape[0]; i++) for (let j = 0; j < shape[1]; j++) for (let k = 0; k < shape[2]; k++) {
    const v = a[i] * b[j] * c[k];
    w[(i * shape[1] + j) * shape[2] + k] = v;
    if (v > max) max = v;
  }
  for (let i = 0; i < w.length; i++) w[i] = Math.max(w[i] / max, 1e-3);
  return w;
}

// The network is 43 MB. It is streamed with a progress readout and kept in the browser's cache, so a second
// visit starts at once.
async function fetchModel(onProgress) {
  let cache = null;
  try {
    cache = await caches.open("brainssl-models-v1");
    const hit = await cache.match(HF_MODEL);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { cache = null; }
  const response = await fetch(HF_MODEL);
  if (!response.ok) throw new Error(`model download failed (${response.status})`);
  const total = Number(response.headers.get("content-length")) || 43.2e6;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  if (cache) {
    try { await cache.put(HF_MODEL, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })); }
    catch { /* storage full or blocked; the next visit downloads again */ }
  }
  return bytes;
}

async function runSegmentation() {
  if (segWorker) { stopSegmentation("Stopped."); return; }
  const out = document.getElementById("seg-out");
  const bar = document.getElementById("seg-progress");
  const caseId = document.getElementById("seg-case").value;
  const index = await getJSON("data/cases/index.json");
  const c = index.cases.find((x) => x.id === caseId);
  if (!hasWebGL2()) {
    out.textContent = "This browser has WebGL2 switched off, and the result is drawn with it. In Chrome or Edge, turn on " +
      "Settings > System > Use graphics acceleration when available and restart the browser; the tumour viewer needs it too.";
    return;
  }
  const gpu = !!navigator.gpu;
  segWorker = new Worker(new URL("./seg-worker.js", import.meta.url), { type: "module" });
  const worker = segWorker;
  document.getElementById("seg-run").innerHTML = 'Stop';
  try {
    out.textContent = "Downloading the model from Hugging Face";
    const model = await fetchModel((got, total) => {
      out.textContent = `Downloading the model from Hugging Face: ${(got / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`;
      if (bar) bar.style.width = `${Math.min(100, Math.round(100 * got / total))}%`;
    });
    if (worker !== segWorker) return;
    out.textContent = "Starting the network";
    if (bar) bar.style.width = "0%";
    await ask(worker, { type: "init", model, ortUrl: ORT, gpu }, [model.buffer]);
    // The viewer volumes are the cropped scan, one 8-bit file per contrast, X fastest.
    const { Niivue } = await import("https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/+esm");
    if (!segViewer) {
      segViewer = new Niivue({ backColor: [0, 0, 0, 1], isColorbar: false, show3Dcrosshair: false, loadingText: "" });
      await segViewer.attachToCanvas(document.getElementById("seg-canvas"));
    }
    await segViewer.loadVolumes(["FLAIR", "T1", "T1Gd", "T2"].map((k) => ({ url: c.files[k], colormap: "gray" })));
    const vols = segViewer.volumes;
    const [nx, ny, nz] = vols[0].hdr.dims.slice(1, 4);
    const n = nx * ny * nz;
    const brain = new Uint8Array(n);
    vols.forEach((v) => { for (let i = 0; i < n; i++) if (v.img[i] > 0) brain[i] = 1; });
    // channels-first (C, X, Y, Z) row-major, z-scored inside the brain
    const image = new Float32Array(4 * n);
    vols.forEach((v, ch) => {
      let s = 0, ss = 0, m = 0;
      for (let i = 0; i < n; i++) if (brain[i]) { s += v.img[i]; ss += v.img[i] * v.img[i]; m++; }
      const mean = s / m, std = Math.sqrt(Math.max(ss / m - mean * mean, 0)) + 1e-6;
      for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
        const src = x + nx * (y + ny * z);
        image[((ch * nx + x) * ny + y) * nz + z] = brain[src] ? (v.img[src] - mean) / std : 0;
      }
    });
    const win = c.window ?? [160, 128, 128];
    const wts = gaussian(win);
    const prob = new Float32Array(3 * n), norm = new Float32Array(n);
    const boxes = [];
    for (const x0 of starts(nx, win[0])) for (const y0 of starts(ny, win[1])) for (const z0 of starts(nz, win[2])) boxes.push([x0, y0, z0]);
    const wn = win[0] * win[1] * win[2];
    let done = 0;
    for (const [x0, y0, z0] of boxes) {
      const tile = new Float32Array(4 * wn);
      for (let ch = 0; ch < 4; ch++) for (let x = 0; x < win[0]; x++) for (let y = 0; y < win[1]; y++) {
        const src = ((ch * nx + x0 + x) * ny + y0 + y) * nz + z0;
        tile.set(image.subarray(src, src + win[2]), ((ch * win[0] + x) * win[1] + y) * win[2]);
      }
      if (worker !== segWorker) return;
      out.textContent = `Window ${done + 1} of ${boxes.length}` +
        (gpu ? "" : " (without WebGPU each window takes about a minute; the page stays usable)");
      const { logits } = await ask(worker, { type: "run", tile, dims: [1, 4, ...win] }, [tile.buffer]);
      for (let r = 0; r < 3; r++) for (let x = 0; x < win[0]; x++) for (let y = 0; y < win[1]; y++) for (let z = 0; z < win[2]; z++) {
        const w = wts[(x * win[1] + y) * win[2] + z];
        const p = 1 / (1 + Math.exp(-logits[((r * win[0] + x) * win[1] + y) * win[2] + z]));
        const dst = (x0 + x) + nx * ((y0 + y) + ny * (z0 + z));
        prob[r * n + dst] += p * w;
        if (r === 0) norm[dst] += w;
      }
      done++;
      bar.style.width = `${Math.round((100 * done) / boxes.length)}%`;
    }
    const labels = new Uint8Array(n);
    let counts = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const wt = prob[i] / norm[i] > 0.5, tc = wt && prob[n + i] / norm[i] > 0.5, et = tc && prob[2 * n + i] / norm[i] > 0.5;
      labels[i] = et ? 3 : tc ? 2 : wt ? 1 : 0;
      if (labels[i]) counts[labels[i] - 1]++;
    }
    const seg = vols[0].clone();
    seg.img = labels;
    seg.hdr.datatypeCode = 2;
    seg.setColormapLabel({ R: [0, 74, 239, 250], G: [0, 222, 68, 204], B: [0, 128, 68, 21], A: [0, 255, 255, 255], I: [0, 1, 2, 3] });
    for (let i = segViewer.volumes.length - 1; i > 0; i--) segViewer.removeVolume(segViewer.volumes[i]);
    segViewer.addVolume(seg);
    segViewer.setOpacity(1, 0.6);
    segViewer.updateGLVolume();
    stopSegmentation();
    out.textContent = `Done: ${counts[0].toLocaleString()} oedema, ${counts[1].toLocaleString()} non-enhancing and ${counts[2].toLocaleString()} enhancing voxels (one pass, no mirror averaging).`;
  } catch (err) {
    if (worker !== segWorker) return;
    stopSegmentation(`Could not run the model here (${err.message ?? err}). The same network runs with the Rust binary or the Docker image.`);
  }
}

export async function initDemo() {
  fmriIndex = await getJSON("data/demo/fmri/index.json");
  const subjSel = document.getElementById("demo-subject"), atlasSel = document.getElementById("demo-atlas");
  if (fmriIndex) {
    subjSel.innerHTML = fmriIndex.subjects.map((s) => `<option value="${s.id}">${s.id} · ${s.site}</option>`).join("");
    atlasSel.innerHTML = Object.keys(fmriIndex.model.atlases).map((a) => `<option value="${a}">${a}</option>`).join("");
    document.getElementById("demo-run").addEventListener("click", runFmri);
    runFmri();   // one subject straight away, so the card never opens empty
  } else {
    document.getElementById("demo-out").innerHTML = `<div class="pending">The demo model is exported with the final results.</div>`;
  }
  const cases = await getJSON("data/cases/index.json");
  if (cases?.cases?.length) {
    document.getElementById("seg-case").innerHTML = cases.cases.map((c) => `<option value="${c.id}">${c.id}</option>`).join("");
    document.getElementById("seg-run").addEventListener("click", runSegmentation);
  }
}
