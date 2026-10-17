// Shell: routing between views, theme, data loading, lazy initialisation.

import { css, d3, dotCI, dumbbell, fmt, lineCI, pairedScatter, registerTable, reliability, renderTable, strip,
         wireTableToggles } from "./charts.js";

const views = ["overview", "viewer", "efficiency", "connectome", "classifiers", "trust-smri", "trust-fmri", "demo", "method",
                "reproduce"];
const cache = new Map();

export async function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return cache.get(path);
}

const initialised = new Set();
const renderers = {};

function pending(el, text = "Results for this panel are being computed.") {
  el.innerHTML = `<div class="pending">${text}</div>`;
}

// ---------------------------------------------------------------- routing

function show(view) {
  if (!views.includes(view)) view = "overview";
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  document.querySelectorAll("[data-view]").forEach((a) => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `brainssl · ${document.querySelector(`[data-view="${view}"]`)?.textContent.trim() ?? ""}`;
  if (!initialised.has(view)) {
    initialised.add(view);
    inits[view]?.();
  } else {
    rerender[view]?.();
  }
}

window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// ---------------------------------------------------------------- theme

document.getElementById("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const current = root.getAttribute("data-theme") ?? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = current === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try { localStorage.setItem("brainssl-theme", next); } catch (e) { /* storage may be blocked */ }
  Object.values(rerender).forEach((f) => f?.());
});

// ---------------------------------------------------------------- overview

let summary = null;

async function initOverview() {
  summary = await getJSON("data/summary.json");
  renderOverview();
  import("./viewer.js").then((m) => m.heroRender(document.getElementById("hero-canvas"), summary?.hero)).catch(() => {});
}

function stat(label, value, unit, delta, foot) {
  const sign = delta?.trim().startsWith("-") || delta?.trim().startsWith("−") ? "down" : "up";
  return `<div class="card stat"><div class="label">${label}</div><div class="value">${value}${unit ? `<small>${unit}</small>` : ""}</div>` +
    (delta ? `<div class="delta"><span class="${sign}">${delta}</span></div>` : "") + (foot ? `<div class="foot">${foot}</div>` : "") + "</div>";
}

function renderOverview() {
  const el = document.getElementById("overview-stats");
  if (!summary) {
    el.innerHTML = "";
    pending(document.getElementById("overview-smri-chart"));
    pending(document.getElementById("overview-fmri-chart"));
    return;
  }
  const s = summary.smri, f = summary.fmri;
  el.innerHTML = [
    s ? stat(`Tumour Dice with ${s.headline_budget} of labels`, fmt(s.ours, 3), "", `${d3.format("+.3f")(s.ours - s.baseline)} vs U-Net from scratch`, `mean of WT, TC and ET over ${s.test_cases} test scans`) : "",
    s?.baseline_scans_to_match ? stat("Annotated scans for the same Dice", `${s.headline_scans}`, ` vs ~${Math.round(s.baseline_scans_to_match)}`, null,
      `ours against the U-Net from scratch, whose curve is interpolated in log(scans)`) : "",
    f ? stat("Autism vs control, AUC", fmt(f.ours, 3), "", `${d3.format("+.3f")(f.ours - f.baseline)} vs ${f.baseline_name}`, `${f.repeats} × ${f.folds}-fold CV, ${f.subjects} subjects`) : "",
    f ? stat("Splits where ours wins", `${f.wins}`, `/ ${f.splits}`, null, `paired by split against ${f.baseline_name}; one-sided p ${pText(f.p)}`) : "",
  ].join("");
  if (s?.curve) {
    lineCI(document.getElementById("overview-smri-chart"), {
      series: [
        { key: "ours", label: "ours", color: css("--series-1"), emphasis: true, points: s.curve.ours },
        { key: "base", label: "U-Net", color: css("--series-2"), points: s.curve.baseline },
      ], xTicks: s.curve.ticks, xLabel: "annotated training scans", yLabel: "mean Dice", height: 230,
    });
  } else pending(document.getElementById("overview-smri-chart"));
  if (f?.ranking) {
    dotCI(document.getElementById("overview-fmri-chart"), {
      rows: f.ranking.map((r) => ({ ...r, color: r.ours ? css("--series-1") : r.main ? css("--series-2") : css("--muted-series"), emphasis: r.ours })),
      xLabel: `AUC, ${f.repeats} × ${f.folds}-fold cross-validation, all five atlases`,
    });
  } else pending(document.getElementById("overview-fmri-chart"));
}

// ---------------------------------------------------------------- label efficiency

let smri = null;
let region = "mean";

async function initEfficiency() {
  smri = await getJSON("data/smri.json");
  document.querySelectorAll("#region-select button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#region-select button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    region = b.dataset.region;
    renderEfficiency();
  }));
  const sel = document.getElementById("paired-budget");
  if (smri) {
    sel.innerHTML = smri.budgets.map((b) => `<option value="${b}" ${b === smri.default_budget ? "selected" : ""}>${b} of labels</option>`).join("");
    sel.addEventListener("change", renderPaired);
  }
  renderEfficiency();
}

function renderEfficiency() {
  if (!smri) {
    ["eff-chart", "paired-chart", "ablation-chart", "hd-table"].forEach((id) => pending(document.getElementById(id)));
    return;
  }
  const key = region === "mean" ? "dice_mean" : `dice_${region}`;
  const pick = (arr) => arr.map((p) => ({ x: p.budget, y: p[key].mean, lo: p[key].lo, hi: p[key].hi }));
  document.getElementById("eff-sub").textContent = `${region === "mean" ? "mean of the three regions" : { WT: "whole tumour", TC: "tumour core", ET: "enhancing tumour" }[region]}, ${smri.test_cases} test scans`;
  renderers["eff-chart"] = () => lineCI(document.getElementById("eff-chart"), {
    series: [
      { key: "ours", label: "Bilateral U-Net, pretrained", color: css("--series-1"), emphasis: true, points: pick(smri.curve.ours) },
      { key: "base", label: "U-Net from scratch", color: css("--series-2"), points: pick(smri.curve.baseline) },
    ], xTicks: smri.budgets, xLabel: "share of the 310 annotated training scans", yLabel: "Dice",
  });
  renderers["eff-chart"]();
  registerTable("eff-chart", [
    { key: "budget", label: "Labels" }, { key: "scans", label: "Scans", num: true },
    { key: "ours", label: "Ours", num: true, format: (v) => fmt(v) }, { key: "baseline", label: "U-Net", num: true, format: (v) => fmt(v) },
    { key: "diff", label: "Difference", num: true, format: (v) => d3.format("+.3f")(v) },
  ], smri.curve.ours.map((p, i) => ({ budget: p.budget, scans: p.scans, ours: p[key].mean, baseline: smri.curve.baseline[i][key].mean, diff: p[key].mean - smri.curve.baseline[i][key].mean })));
  renderPaired();
  renderAblation();
  const t = smri.hd95;
  renderTable(document.getElementById("hd-table"), [
    { key: "budget", label: "Labels" }, { key: "model", label: "Model" },
    { key: "WT", label: "WT", num: true, format: (v) => fmt(v, 1) }, { key: "TC", label: "TC", num: true, format: (v) => fmt(v, 1) },
    { key: "ET", label: "ET", num: true, format: (v) => fmt(v, 1) },
  ], t.map((r) => ({ ...r, _ours: r.ours })));
}

function renderPaired() {
  if (!smri) return;
  const budget = document.getElementById("paired-budget").value || smri.default_budget;
  const key = region === "mean" ? "dice_mean" : `dice_${region}`;
  const cases = smri.per_case[budget] ?? [];
  pairedScatter(document.getElementById("paired-chart"), {
    points: cases.map((c) => ({ id: c.case_id, x: c.baseline[key], y: c.ours[key] })),
    xLabel: "U-Net from scratch", yLabel: "Bilateral U-Net, pretrained", color: css("--series-1"),
  });
}

function renderAblation() {
  if (!smri?.ablation) { pending(document.getElementById("ablation-chart")); return; }
  const rows = smri.ablation.map((r) => ({ label: r.label, value: r.mean, lo: r.lo, hi: r.hi, emphasis: r.ours,
    color: r.ours ? css("--series-1") : r.baseline ? css("--series-2") : css("--series-3") }));
  renderers["ablation-chart"] = () => dotCI(document.getElementById("ablation-chart"), { rows, xLabel: "mean Dice, 10% of labels" });
  renderers["ablation-chart"]();
  registerTable("ablation-chart", [{ key: "label", label: "Variant" }, { key: "value", label: "Dice", num: true, format: (v) => fmt(v) },
    { key: "lo", label: "95% low", num: true, format: (v) => fmt(v) }, { key: "hi", label: "95% high", num: true, format: (v) => fmt(v) }], rows.map((r) => ({ ...r, _ours: r.emphasis })));
}

// ---------------------------------------------------------------- classifiers

let fmri = null;
let metric = "auc";
let clfView = "ensemble";
const METRIC_LABEL = { auc: "AUC", accuracy: "accuracy", balanced_accuracy: "balanced accuracy" };
const pText = (p) => (p == null ? "–" : p < 0.001 ? "< 0.001" : d3.format(".3f")(p));

async function initClassifiers() {
  fmri = await getJSON("data/fmri.json");
  document.querySelectorAll("#clf-metric button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#clf-metric button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    metric = b.dataset.metric;
    renderClassifiers();
  }));
  const sel = document.getElementById("clf-atlas");
  if (fmri) {
    sel.innerHTML = fmri.views.map((v) => `<option value="${v.id}">${v.label}</option>`).join("");
    sel.addEventListener("change", () => { clfView = sel.value; renderClassifiers(); });
  }
  renderClassifiers();
}

function renderClassifiers() {
  const ids = ["atlas-chart", "clf-chart", "folds-chart", "site-chart", "calib-chart"];
  if (!fmri) { ids.forEach((id) => pending(document.getElementById(id))); return; }
  const label = METRIC_LABEL[metric];
  const view = fmri.views.find((v) => v.id === clfView) ?? fmri.views[0];

  // every atlas: ours against the strongest baseline on that atlas
  const atlasRows = fmri.views.map((v) => {
    const ours = v.rows.find((r) => r.ours), base = v.rows.find((r) => r.key === v.main);
    return { label: v.id === "ensemble" ? "all five" : v.id, a: ours[metric].mean, b: base[metric].mean,
      base: base.label, p: v.tests[metric].p_one_sided, wins: v.tests[metric].wins, splits: v.tests[metric].fold_diffs.length };
  });
  document.getElementById("atlas-sub").textContent =
    `mean ${label} over ${fmri.repeats} repeats; the comparison on each row is the baseline that scores best on that atlas`;
  renderers["atlas-chart"] = () => dumbbell(document.getElementById("atlas-chart"), {
    rows: atlasRows, aLabel: "noise-aware (ours)", bLabel: "strongest baseline", aColor: css("--series-1"), bColor: css("--series-2"),
    xLabel: label, tip: (r) => [r.label, [["ours", fmt(r.a)], [r.base, fmt(r.b)], ["difference", d3.format("+.3f")(r.a - r.b)],
      ["splits won", `${r.wins} of ${r.splits}`], ["one-sided p", pText(r.p)]]],
  });
  renderers["atlas-chart"]();
  registerTable("atlas-chart", [{ key: "label", label: "Atlas" }, { key: "a", label: "Ours", num: true, format: (v) => fmt(v) },
    { key: "base", label: "Strongest baseline" }, { key: "b", label: "Baseline", num: true, format: (v) => fmt(v) },
    { key: "wins", label: "Splits won", num: true }, { key: "p", label: "p", num: true, format: pText }], atlasRows);

  // all models on the selected atlas
  const rows = view.rows.map((r) => ({ label: r.label, value: r[metric].mean, lo: r[metric].lo, hi: r[metric].hi, emphasis: r.ours,
    color: r.ours ? css("--series-1") : r.key === view.main ? css("--series-2") : css("--muted-series"),
    extra: [["std over repeats", fmt(r[metric].std)], ...(r.C != null ? [["C", d3.format("~g")(r.C)]] : [])] }));
  document.getElementById("clf-sub").textContent = `${view.label}: mean over ${fmri.repeats} repeats with 95% bootstrap interval`;
  renderers["clf-chart"] = () => dotCI(document.getElementById("clf-chart"), { rows, xLabel: label });
  renderers["clf-chart"]();
  registerTable("clf-chart", [{ key: "label", label: "Model" }, { key: "value", label, num: true, format: (v) => fmt(v) },
    { key: "lo", label: "95% low", num: true, format: (v) => fmt(v) }, { key: "hi", label: "95% high", num: true, format: (v) => fmt(v) }],
    rows.map((r) => ({ ...r, _ours: r.emphasis })));

  const t = view.tests[metric];
  document.getElementById("folds-sub").textContent =
    `ours minus ${view.main_label} on each of the ${t.fold_diffs.length} splits: ${t.wins} won, ${t.losses} lost; ` +
    `corrected resampled t-test, one-sided p ${pText(t.p_one_sided)}`;
  strip(document.getElementById("folds-chart"), { values: t.fold_diffs, label: `difference in ${label}`, color: css("--series-1") });

  if (fmri.sites) {
    const siteRows = fmri.sites.map((s) => ({ label: s.site, a: s.ours, b: s.baseline, n: s.n }));
    renderers["site-chart"] = () => dumbbell(document.getElementById("site-chart"), { rows: siteRows, aLabel: "ours",
      bLabel: fmri.site_baseline, aColor: css("--series-1"), bColor: css("--series-2"), xLabel: "accuracy on the held-out site" });
    renderers["site-chart"]();
    registerTable("site-chart", [{ key: "label", label: "Site" }, { key: "n", label: "Subjects", num: true },
      { key: "a", label: "Ours", num: true, format: (v) => fmt(v) }, { key: "b", label: fmri.site_baseline, num: true, format: (v) => fmt(v) }], siteRows);
  } else pending(document.getElementById("site-chart"));

  if (fmri.calibration) {
    reliability(document.getElementById("calib-chart"), { series: [
      { label: "ours", color: css("--series-1"), points: fmri.calibration.ours },
      { label: fmri.calibration_baseline, color: css("--series-2"), points: fmri.calibration.baseline },
    ] });
  } else pending(document.getElementById("calib-chart"));
}

// ---------------------------------------------------------------- reproduce

async function initReproduce() {
  document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => {
    const text = b.parentElement.textContent.trim();
    navigator.clipboard?.writeText(text);
    b.title = "Copied";
  }));
  const s = summary ?? await getJSON("data/summary.json");
  const env = s?.environment;
  if (env) document.getElementById("env-kv").innerHTML = Object.entries(env).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  import("./explainers.js").then((m) => m.pipelineDag(document.getElementById("dag")));
}

// ---------------------------------------------------------------- wiring

const inits = {
  overview: initOverview,
  viewer: () => import("./viewer.js").then((m) => m.initViewer()),
  efficiency: initEfficiency,
  connectome: () => import("./connectome.js").then((m) => m.initConnectome()),
  classifiers: initClassifiers,
  "trust-smri": () => import("./trust.js").then((m) => m.initTrustSmri()),
  "trust-fmri": () => import("./trust.js").then((m) => m.initTrustFmri()),
  demo: () => import("./demo.js").then((m) => m.initDemo()),
  method: () => import("./explainers.js").then((m) => m.initExplainers()),
  reproduce: initReproduce,
};
const rerender = {
  overview: () => summary && renderOverview(),
  efficiency: () => smri && renderEfficiency(),
  classifiers: () => fmri && renderClassifiers(),
  "trust-smri": () => import("./trust.js").then((m) => m.redrawTrustSmri()),
  "trust-fmri": () => import("./trust.js").then((m) => m.redrawTrustFmri()),
  method: () => import("./explainers.js").then((m) => m.initExplainers()),
};

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const active = document.querySelector(".view.active")?.id.replace("view-", "");
    rerender[active]?.();
  }, 200);
});

wireTableToggles(renderers);
show(location.hash.slice(1) || "overview");
