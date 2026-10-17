// Tumour viewer and the rotating render on the overview page, both on NiiVue.

import { Niivue, SLICE_TYPE } from "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/+esm";
import { getJSON } from "./app.js";
import { fmt } from "./charts.js";

// BraTS colour convention: oedema green, non-enhancing core red, enhancing yellow.
const LABELS = {
  R: [0, 74, 239, 250], G: [0, 222, 68, 204], B: [0, 128, 68, 21], A: [0, 255, 255, 255], I: [0, 1, 2, 3],
  labels: ["background", "oedema", "non-enhancing core", "enhancing tumour"],
};
// Errors of a prediction against the reference: 1 missed tumour, 2 false alarm.
const ERRORS = {
  R: [0, 213, 34], G: [0, 81, 211], B: [0, 129, 238], A: [0, 255, 255], I: [0, 1, 2],
  labels: ["agree", "missed tumour", "false alarm"],
};

const viewerOpts = {
  backColor: [0, 0, 0, 1], show3Dcrosshair: false, isColorbar: false, isOrientCube: false,
  crosshairColor: [0.42, 0.69, 1, 0.85], crosshairWidth: 1, multiplanarShowRender: 2,
  dragAndDropEnabled: false, scrollRequiresFocus: true, isResizeCanvas: true, loadingText: "",
  clipPlaneColor: [0, 0, 0, 0],   // NiiVue tints the cut face magenta by default
};

function overlayFile(c, overlay) {
  return { ours: c.files.ours, baseline: c.files.baseline, truth: c.files.truth, errors: c.files.errors,
           uncertainty: c.files.uncertainty, saliency: c.files.saliency }[overlay];
}

// ---------------------------------------------------------------- hero

export async function heroRender(canvas, hero) {
  const index = await getJSON("data/cases/index.json");
  const c = index?.cases?.find((x) => x.id === (hero ?? index?.hero)) ?? index?.cases?.[0];
  if (!c) return;
  const nv = new Niivue({ ...viewerOpts, scrollRequiresFocus: true });
  await nv.attachToCanvas(canvas);
  await nv.loadVolumes([
    { url: c.files.FLAIR, colormap: "gray", opacity: 0.85 },
    { url: c.files.truth, colormap: "gray", opacity: 1 },
  ]);
  nv.volumes[1].setColormapLabel(LABELS);
  nv.updateGLVolume();
  nv.setSliceType(SLICE_TYPE.RENDER);
  nv.setClipPlane([0.05, 270, 0]);
  document.getElementById("hero-case").textContent = `${c.id} · reference annotation`;
  let az = 110;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const spin = () => {
    if (!document.getElementById("view-overview").classList.contains("active")) { requestAnimationFrame(spin); return; }
    az = (az + 0.18) % 360;
    nv.setRenderAzimuthElevation(az, 18);
    requestAnimationFrame(spin);
  };
  if (!reduce) requestAnimationFrame(spin);
  else nv.setRenderAzimuthElevation(110, 18);
}

// ---------------------------------------------------------------- viewer

let nv = null;
let index = null;
let current = null;

const state = { contrast: "FLAIR", overlay: "ours", opacity: 0.6, mode: SLICE_TYPE.MULTIPLANAR, asymmetry: false };

async function loadCase(c) {
  current = c;
  document.querySelectorAll(".case-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === c.id)));
  document.getElementById("viewer-badge").textContent = `${c.id} · loading`;
  const list = [{ url: c.files[state.contrast], colormap: "gray" }];
  const ov = overlayFile(c, state.overlay);
  const continuous = state.overlay === "uncertainty" || state.overlay === "saliency";
  if (ov) list.push(continuous ? { url: ov, colormap: "inferno", opacity: state.opacity, cal_min: 8, cal_max: 200 }
                               : { url: ov, colormap: "gray", opacity: state.opacity });
  if (state.asymmetry && c.files.asymmetry) list.push({ url: c.files.asymmetry, colormap: "warm", opacity: 0.55, cal_min: 40, cal_max: 200 });
  await nv.loadVolumes(list);
  if (ov && !continuous) nv.volumes[1].setColormapLabel(state.overlay === "errors" ? ERRORS : LABELS);
  nv.updateGLVolume();
  nv.setSliceType(state.mode);
  document.getElementById("viewer-badge").textContent = `${c.id} · ${state.contrast}`;
  renderLegend();
  renderMetrics(c);
}

function renderLegend() {
  const el = document.getElementById("viewer-legend");
  if (state.overlay === "errors") {
    el.innerHTML = `<span><i class="swatch" style="background:#d55181"></i>missed tumour (false negative)</span>
      <span><i class="swatch" style="background:#22d3ee"></i>false alarm (false positive)</span>`;
  } else if (state.overlay === "uncertainty" || state.overlay === "saliency") {
    const what = state.overlay === "uncertainty" ? "entropy of the whole-tumour probability, 0 to 0.69 nats" : "gradient × input of the whole-tumour logit";
    el.innerHTML = `<span><i class="swatch" style="background:linear-gradient(90deg,#1b0c41,#cf4446,#fcffa4)"></i>${what}, low to high</span>`;
  } else if (state.overlay === "none") {
    el.innerHTML = "";
  } else {
    el.innerHTML = `<span><i class="swatch" style="background:var(--label-oedema)"></i>oedema</span>
      <span><i class="swatch" style="background:var(--label-core)"></i>non-enhancing core</span>
      <span><i class="swatch" style="background:var(--label-enhancing)"></i>enhancing tumour</span>`;
  }
  if (state.asymmetry) el.innerHTML += `<span><i class="swatch" style="background:linear-gradient(90deg,#7a1d00,#ffd84a)"></i>left-right asymmetry of FLAIR</span>`;
}

function renderMetrics(c) {
  const names = { WT: "whole tumour", TC: "tumour core", ET: "enhancing" };
  const rows = [
    ...["WT", "TC", "ET"].map((r) => ({ metric: `Dice, ${names[r]}`, ours: c.dice.ours[r], base: c.dice.baseline[r], digits: 3, higher: true })),
    ...["WT", "TC", "ET"].map((r) => ({ metric: `HD95, ${names[r]}`, ours: c.hd95.ours[r], base: c.hd95.baseline[r], digits: 1, higher: false })),
  ];
  const wins = (r, side) => {
    const [a, b] = side === "ours" ? [r.ours, r.base] : [r.base, r.ours];
    return r.higher ? a > b : a < b;
  };
  const cell = (r, side) => `<td class="num${wins(r, side) ? " best" : ""}">${fmt(r[side], r.digits)}</td>`;
  document.getElementById("case-metrics").innerHTML = `<div class="table-scroll"><table><thead><tr><th></th>` +
    `<th class="num">Ours</th><th class="num">U-Net</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${r.metric}</td>${cell(r, "ours")}${cell(r, "base")}</tr>`).join("") + "</tbody></table></div>";
  document.getElementById("case-note").textContent = c.note ?? "";
}

export async function initViewer() {
  index = await getJSON("data/cases/index.json");
  const list = document.getElementById("case-list");
  if (!index?.cases?.length) {
    list.innerHTML = `<div class="pending">Test scans are added once the final models are trained.</div>`;
    document.getElementById("viewer-badge").textContent = "no scans yet";
    return;
  }
  if (!index.cases.some((c) => c.files.uncertainty)) {
    document.querySelectorAll("#overlay option[data-trust]").forEach((o) => o.remove());
  }
  list.innerHTML = index.cases.map((c) => `<button type="button" class="case-btn" data-id="${c.id}" aria-pressed="false">
      <div class="id">${c.id}</div><div class="meta"><span>WT ${fmt(c.dice.ours.WT, 2)} vs ${fmt(c.dice.baseline.WT, 2)}</span><span>${c.tag ?? ""}</span></div></button>`).join("");
  nv = new Niivue(viewerOpts);
  await nv.attachToCanvas(document.getElementById("viewer-canvas"));

  list.querySelectorAll(".case-btn").forEach((b) => b.addEventListener("click", () => loadCase(index.cases.find((c) => c.id === b.dataset.id))));
  document.getElementById("contrast").addEventListener("change", (e) => { state.contrast = e.target.value; loadCase(current); });
  document.getElementById("overlay").addEventListener("change", (e) => { state.overlay = e.target.value; loadCase(current); });
  document.getElementById("mirror").addEventListener("change", (e) => { state.asymmetry = e.target.checked; loadCase(current); });
  document.getElementById("opacity").addEventListener("input", (e) => {
    state.opacity = Number(e.target.value);
    if (nv.volumes.length > 1) nv.setOpacity(1, state.opacity);
  });
  document.querySelectorAll("#view-mode button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#view-mode button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.mode = Number(b.dataset.mode);
    nv.setSliceType(state.mode);
    if (state.mode === SLICE_TYPE.RENDER) nv.setClipPlane([0.05, 270, 0]);
    else nv.setClipPlane([2, 0, 0]);
  }));
  await loadCase(index.cases[0]);
}
