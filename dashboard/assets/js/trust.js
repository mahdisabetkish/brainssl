// Trust audits: a scorecard per model and the evidence behind each line.
// Two views share this module: the connectome classifier and the segmentation network.

import { css, d3, dotCI, dumbbell, hideTip, reliability, renderTable, showTip } from "./charts.js";
import { getJSON } from "./app.js";

const ok = { holds: "holds", partly: "partly", fails: "does not hold" };
const ICON = { holds: "✓", partly: "◐", fails: "✕" };
const f2 = d3.format(".2f"), f3 = d3.format(".3f");
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

let trust = null;
async function load() {
  trust ??= await getJSON("data/trust.json");
  return trust;
}

function pending(el, text = "These checks run after the final models are trained.") {
  el.innerHTML = `<div class="pending">${text}</div>`;
}

// ---------------------------------------------------------------- scorecard

function scorecard(el, checks) {
  el.innerHTML = checks.map((c) => `
    <div class="check ${c.status}" tabindex="0">
      <div class="check-top"><span class="badge ${c.status}" aria-hidden="true">${ICON[c.status]}</span>
        <span class="verdict">${ok[c.status]}</span></div>
      <div class="check-name">${c.name}</div>
      <div class="check-evidence">${c.evidence}</div>
      <div class="check-rule">Rule: ${c.rule}</div>
    </div>`).join("");
}

const grade = (holds, partly) => (holds ? "holds" : partly ? "partly" : "fails");

// ---------------------------------------------------------------- generic charts

function frame(el, height, margin) {
  el.innerHTML = "";
  const w = Math.max(320, el.clientWidth || 600);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${height}`).attr("role", "img");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { g, iw: w - margin.left - margin.right, ih: height - margin.top - margin.bottom };
}

function legendRow(el, items) {
  const div = document.createElement("div");
  div.className = "legend";
  div.style.marginBottom = "8px";
  div.innerHTML = items.map((i) => `<span><i class="swatch" style="background:${i.color}"></i>${i.label}</span>`).join("");
  el.prepend(div);
}

function axes(g, x, y, iw, ih, xLabel, yLabel, xFormat) {
  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(xFormat ?? null).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 38).attr("text-anchor", "middle").text(xLabel);
  g.append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -42).attr("text-anchor", "middle").text(yLabel);
}

/** Lines over a numeric x; series: [{label, color, points: [{x, y}], dashed, width}] */
function lines(el, { series, xLabel, yLabel, xDomain, yDomain, height = 300, tipTitle = "x", xFormat }) {
  const { g, iw, ih } = frame(el, height, { top: 12, right: 20, bottom: 46, left: 56 });
  const all = series.flatMap((s) => s.points);
  const x = d3.scaleLinear().domain(xDomain ?? d3.extent(all, (p) => p.x)).range([0, iw]);
  const y = d3.scaleLinear().domain(yDomain ?? d3.extent(all, (p) => p.y)).nice().range([ih, 0]);
  axes(g, x, y, iw, ih, xLabel, yLabel, xFormat);
  const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y));
  series.forEach((s) => {
    g.append("path").datum(s.points).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.width ?? 2)
      .attr("stroke-dasharray", s.dashed ? "5 4" : null).attr("stroke-linejoin", "round").attr("d", line);
    if (!s.dashed && s.points.length < 30) {
      g.append("g").selectAll("circle").data(s.points).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 3.5)
        .attr("fill", s.color).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5);
    }
  });
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [mx] = d3.pointer(event);
      const target = x.invert(mx);
      const xv = xs.reduce((best, v) => (Math.abs(v - target) < Math.abs(best - target) ? v : best), xs[0]);
      showTip(event, `${tipTitle} ${xFormat ? xFormat(xv) : f2(xv)}`, series.map((s) => {
        const p = s.points.find((q) => q.x === xv);
        return [s.label, p ? f3(p.y) : "–"];
      }));
    })
    .on("mouseleave", hideTip);
  legendRow(el, series.map((s) => ({ label: s.label, color: s.color })));
}

/** Scatter with an identity line and optional log axes; points: [{x, y, id}] */
function scatter(el, { points, xLabel, yLabel, color, log = false, identity = false, height = 320, note, xFormat }) {
  const { g, iw, ih } = frame(el, height, { top: 12, right: 18, bottom: 46, left: 56 });
  const pos = (v) => (log ? Math.max(v, 0.5) : v);
  const ex = d3.extent(points, (p) => pos(p.x)), ey = d3.extent(points, (p) => pos(p.y));
  const lo = identity ? Math.min(ex[0], ey[0]) : null, hi = identity ? Math.max(ex[1], ey[1]) : null;
  const make = (ext, a, b) => (log ? d3.scaleLog() : d3.scaleLinear()).domain(identity ? [a, b] : ext).nice().range([0, 1]);
  const x = make(ex, lo, hi).range([0, iw]);
  const y = make(ey, lo, hi).range([ih, 0]);
  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5, log ? "~g" : undefined).tickFormat(xFormat ?? null).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, log ? "~g" : undefined).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 38).attr("text-anchor", "middle").text(xLabel);
  g.append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -42).attr("text-anchor", "middle").text(yLabel);
  if (identity) {
    const [a, b] = x.domain();
    g.append("line").attr("x1", x(a)).attr("y1", y(a)).attr("x2", x(b)).attr("y2", y(b)).attr("stroke", css("--axis"));
  }
  g.append("g").selectAll("circle").data(points).join("circle")
    .attr("cx", (p) => x(pos(p.x))).attr("cy", (p) => y(pos(p.y))).attr("r", 4)
    .attr("fill", color).attr("fill-opacity", 0.8).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5);
  if (note) g.append("text").attr("class", "label-strong").attr("x", 8).attr("y", 14).text(note);
  const del = d3.Delaunay.from(points, (p) => x(pos(p.x)), (p) => y(pos(p.y)));
  g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [mx, my] = d3.pointer(event);
      const p = points[del.find(mx, my)];
      if (p) showTip(event, p.id, [[xLabel, p.xs ?? f3(p.x)], [yLabel, p.ys ?? f3(p.y)]]);
    })
    .on("mouseleave", hideTip);
}

/** Heat map; values[r][c] with row and column labels, a diverging or sequential scale. */
function heatmap(el, { rows, cols, values, format = f3, scale = "seq", label, cell = 38, domain }) {
  const margin = { top: 52, right: 12, bottom: 8, left: Math.min(150, Math.max(...rows.map((r) => r.length)) * 7 + 14) };
  const h = margin.top + margin.bottom + rows.length * cell;
  const { g, iw } = frame(el, h, margin);
  const flat = values.flat().filter((v) => v != null && Number.isFinite(v));
  const max = d3.max(flat.map(Math.abs)) || 1;
  const color = scale === "div"
    ? d3.scaleDiverging([-max, 0, max], (t) => d3.interpolateRgbBasis([css("--div-neg"), css("--div-mid"), css("--div-pos")])(t))
    : d3.scaleSequential(domain ?? [0, d3.max(flat) || 1], d3.interpolateRgbBasis([css("--surface-3"), css("--series-4"), css("--series-2")]));
  const cw = Math.min(cell + 14, iw / cols.length);
  const x = (j) => j * cw;
  rows.forEach((r, i) => {
    g.append("text").attr("class", "label").attr("x", -8).attr("y", i * cell + cell / 2 + 4).attr("text-anchor", "end").text(r);
    cols.forEach((c, j) => {
      const v = values[i][j];
      g.append("rect").attr("x", x(j)).attr("y", i * cell).attr("width", cw - 3).attr("height", cell - 3).attr("rx", 5)
        .attr("fill", v == null || !Number.isFinite(v) ? css("--surface-3") : color(v))
        .on("mousemove", (event) => showTip(event, `${r} · ${c}`, [[label, v == null ? "–" : format(v)]]))
        .on("mouseleave", hideTip);
      if (v != null && Number.isFinite(v) && cw >= 40) {
        const lum = d3.hsl(color(v)).l;
        g.append("text").attr("x", x(j) + (cw - 3) / 2).attr("y", i * cell + cell / 2 + 3).attr("text-anchor", "middle")
          .attr("fill", lum > 0.55 ? "#0b0d12" : "#f2f4f8").style("font-size", "11px").style("pointer-events", "none").text(format(v));
      }
    });
  });
  cols.forEach((c, j) => {
    g.append("text").attr("class", "label").attr("transform", `translate(${x(j) + (cw - 3) / 2},-10) rotate(-35)`).text(c);
  });
}

// ================================================================ connectome classifier

const MODEL_ORDER = ["vicinal", "tangent", "log", "corr"];
const MODEL_NAME = { vicinal: "Noise-aware (ours)", tangent: "Tangent space + LR", log: "Log-Euclidean + LR", corr: "Correlation + LR" };
const modelColor = (m) => (m === "vicinal" ? css("--series-1") : m === "log" ? css("--series-2") : m === "tangent" ? css("--series-3") : css("--series-4"));

const state = { calModel: "vicinal", calScheme: "kfold", group: "site", atlas: "aal", blockKind: "mean_abs", smriBudget: "10pct", smriRegion: "WT" };

function fmriChecks(d) {
  const k = d.fmri?.kfold, i = d.interpret;
  if (!k) return [];
  const rec = k.recalibration.vicinal, sc = k.shortcuts, sel = k.selective.vicinal;
  const checks = [];
  const calOk = (r) => Math.abs(r.slope.mean - 1) <= 0.25 && r.ece.mean <= 0.05;
  checks.push({
    name: "Probabilities mean what they say",
    status: calOk(rec.raw) ? "holds" : calOk(rec.recalibrated) ? "partly" : "fails",
    evidence: `Raw slope ${f2(rec.raw.slope.mean)}, ECE ${f3(rec.raw.ece.mean)}; after recalibration slope ${f2(rec.recalibrated.slope.mean)}, ECE ${f3(rec.recalibrated.ece.mean)}.`,
    rule: "slope 0.75 to 1.25 and ECE ≤ 0.05; partly if only after recalibration",
  });
  const auc = sc.model.auc.mean, site = sc.nuisance_only.find((r) => r.inputs === "site").auc.mean, motion = sc.nuisance_only.find((r) => r.inputs === "motion").auc.mean;
  const withinSite = sc.model.auc_within_site.mean, withinMotion = sc.model.auc_within_motion_tercile.mean;
  const siteScore = [withinSite >= auc - 0.02, site <= auc - 0.10].filter(Boolean).length;
  checks.push({
    name: "Not riding on the site", status: siteScore === 2 ? "holds" : siteScore === 1 ? "partly" : "fails",
    evidence: `Site alone gives AUC ${f2(site)}; the model gives ${f2(auc)} overall and ${f2(withinSite)} inside sites.`,
    rule: "within-site AUC ≥ overall − 0.02 and site-only AUC ≤ overall − 0.10",
  });
  const motionScore = [motion <= auc - 0.10, withinMotion >= auc - 0.03].filter(Boolean).length;
  checks.push({
    name: "Not riding on head motion", status: motionScore === 2 ? "holds" : motionScore === 1 ? "partly" : "fails",
    evidence: `Motion alone gives AUC ${f2(motion)}; inside motion terciles the model keeps ${f2(withinMotion)}.`,
    rule: "motion-only AUC ≤ overall − 0.10 and AUC inside terciles ≥ overall − 0.03",
  });
  const gain = sel.accuracy_at_50 - sel.accuracy_full;
  checks.push({
    name: "Knows when it is unsure", status: grade(gain >= 0.05, gain >= 0.02),
    evidence: `Accuracy ${f2(sel.accuracy_full)} on all subjects, ${f2(sel.accuracy_at_50)} on the most confident half.`,
    rule: "accuracy on the confident half ≥ overall + 0.05 (partly: +0.02)",
  });
  const s = k.subgroups.vicinal.site_summary, frac = s.sites_scored ? s.sites_ci_above_half / s.sites_scored : 0;
  checks.push({
    name: "Holds in every group", status: grade(frac >= 0.8, frac >= 0.6),
    evidence: `${s.sites_ci_above_half} of ${s.sites_scored} scorable sites have an interval above 0.5; worst site AUC ${f2(s.worst_auc)}.`,
    rule: "interval excludes 0.5 for ≥ 80% of scorable sites (partly: 60%)",
  });
  if (i?.deletion?.aal) {
    const del = i.deletion.aal, r = del.fractions["0.05"];
    const dropS = del.baseline_auc - r.supportive, dropR = Math.max(1e-6, del.baseline_auc - r.random);
    checks.push({
      name: "The explanation is faithful", status: grade(dropS >= 3 * dropR, dropS >= 2 * dropR),
      evidence: `Deleting 5% of the most supportive features: AUC ${f2(del.baseline_auc)} → ${f2(r.supportive)}; random features: → ${f2(r.random)}.`,
      rule: "drop ≥ 3 × the drop from random deletion (partly: 2×)",
    });
  }
  if (i?.stability?.aal) {
    const st = i.stability.aal;
    checks.push({
      name: "The explanation is stable", status: grade(st.node_strength_rank_correlation >= 0.6, st.node_strength_rank_correlation >= 0.3),
      evidence: `Region strengths agree between refits at rank correlation ${f2(st.node_strength_rank_correlation)}; the same 100 strongest features overlap only ${f2(st.top_k_jaccard_pairs)}.`,
      rule: "node-strength rank correlation ≥ 0.6 (partly: 0.3)",
    });
  }
  return checks;
}

function renderCalibration(d) {
  const block = d.fmri[state.calScheme];
  const m = state.calModel;
  const cal = block.calibration[m], rec = block.recalibration[m];
  const pts = (bins) => bins.map((b) => ({ x: b.mean_pred, y: b.observed, n: b.n }));
  reliability(document.getElementById("t-cal-chart"), { series: [
    { label: "raw probabilities", color: css("--series-2"), points: pts(cal.bins) },
    { label: "recalibrated", color: css("--series-1"), points: pts(rec.bins_recalibrated) },
  ] });
  document.getElementById("t-cal-note").innerHTML =
    `${MODEL_NAME[m]}: slope ${f2(rec.raw.slope.mean)} → ${f2(rec.recalibrated.slope.mean)}, ` +
    `ECE ${f3(rec.raw.ece.mean)} → ${f3(rec.recalibrated.ece.mean)}, Brier ${f3(rec.raw.brier.mean)} → ${f3(rec.recalibrated.brier.mean)}. ` +
    `Mean confidence ${f2(cal.mean_confidence)} against accuracy ${f2(cal.accuracy)}: ` +
    (cal.mean_confidence < cal.accuracy ? "the raw predictions are under-confident." : "the raw predictions are over-confident.");
  const rows = MODEL_ORDER.filter((x) => block.recalibration[x]).map((x) => {
    const r = block.recalibration[x];
    return { model: MODEL_NAME[x], ece: r.raw.ece.mean, slope: r.raw.slope.mean, ece2: r.recalibrated.ece.mean, slope2: r.recalibrated.slope.mean,
      bacc: r.raw.balanced_accuracy.mean, bacc2: r.recalibrated.balanced_accuracy.mean, _ours: x === "vicinal" };
  });
  renderTable(document.getElementById("t-cal-table"), [
    { key: "model", label: "Model" }, { key: "ece", label: "ECE", num: true, format: f3 }, { key: "slope", label: "slope", num: true, format: f2 },
    { key: "ece2", label: "ECE, recal.", num: true, format: f3 }, { key: "slope2", label: "slope, recal.", num: true, format: f2 },
    { key: "bacc", label: "bal. acc.", num: true, format: f3 }, { key: "bacc2", label: "bal. acc., recal.", num: true, format: f3 },
  ], rows);
}

function renderSelective(d) {
  const sel = d.fmri.kfold.selective;
  const series = MODEL_ORDER.filter((m) => sel[m]).map((m) => ({
    label: MODEL_NAME[m], color: modelColor(m), width: m === "vicinal" ? 3 : 1.8,
    points: sel[m].coverage.map((c, i) => ({ x: c, y: 1 - sel[m].risk[i] })),
  }));
  const o = sel.vicinal;
  series.push({ label: "ideal ranking of cases", color: css("--axis"), dashed: true, points: o.coverage.map((c, i) => ({ x: c, y: 1 - o.oracle_risk[i] })) });
  lines(document.getElementById("t-sel-chart"), { series, xLabel: "fraction of subjects scored (most confident first)", yLabel: "accuracy",
    xDomain: [0, 1], yDomain: [0.6, 1], tipTitle: "coverage", xFormat: d3.format(".0%") });
  document.getElementById("t-sel-note").textContent =
    `Ours: ${f2(o.accuracy_full)} on all subjects, ${f2(o.accuracy_at_80)} on the most confident 80%, ${f2(o.accuracy_at_50)} on the most confident half. ` +
    `Excess risk ${f2(o.excess.mean)} (0 would be a perfect ranking, 1 no better than chance).`;
}

function renderShortcuts(d) {
  const sc = d.fmri.kfold.shortcuts, base = sc.baseline_within_site ?? {};
  const rows = [
    ...sc.nuisance_only.map((r) => ({ label: `only ${r.inputs}`, value: r.auc.mean, color: css("--muted-series") })),
    { label: "connectome model, everyone", value: sc.model.auc.mean, color: css("--series-1"), emphasis: true },
    { label: "… inside sites", value: sc.model.auc_within_site.mean, color: css("--series-1") },
    { label: "… inside motion terciles", value: sc.model.auc_within_motion_tercile.mean, color: css("--series-1") },
    { label: "… inside site and motion", value: sc.model.auc_within_site_and_motion.mean, color: css("--series-1") },
  ];
  dotCI(document.getElementById("t-short-chart"), { rows, xLabel: "AUC (0.5 is chance)", domain: [0.4, 0.9], reference: 0.5 });
  const rho = (r) => `${f2(r.rho)} [${f2(r.ci_low)}, ${f2(r.ci_high)}]`;
  document.getElementById("t-short-note").textContent =
    `Rank correlation of the score with head motion: controls ${rho(sc.model.score_vs_motion_controls)}, autism ${rho(sc.model.score_vs_motion_autism)}.` +
    (base.vicinal ? ` Baseline within-site AUC: ${["log", "tangent", "corr"].filter((m) => base[m]).map((m) => `${MODEL_NAME[m]} ${f2(base[m].mean)}`).join(", ")}.` : "");
}

function renderGroups(d) {
  const sub = d.fmri.kfold.subgroups.vicinal[state.group];
  const rows = sub.filter((r) => r.auc != null).map((r) => ({ label: `${r.group} (n=${r.n})`, value: r.auc, lo: r.ci_low, hi: r.ci_high,
    color: r.ci_low > 0.5 ? css("--series-1") : css("--series-2") }));
  const skipped = sub.filter((r) => r.auc == null);
  dotCI(document.getElementById("t-group-chart"), { rows, xLabel: "AUC of the noise-aware model, 95% bootstrap interval", domain: [0, 1], reference: 0.5 });
  document.getElementById("t-group-note").textContent = skipped.length
    ? `Not scored (fewer than 8 subjects in a class): ${skipped.map((r) => `${r.group} (n=${r.n})`).join(", ")}. Orange intervals include 0.5.`
    : "Orange intervals include 0.5.";
}

function renderWeights(d) {
  const w = d.interpret.weights[state.atlas];
  const rows = w.top_rois.map((r) => ({ label: r.name, value: r.strength, color: css("--series-1"),
    extra: [["signed sum", f3(r.signed)], ["MNI x, y, z", r.mni_mm.join(", ")]] }));
  dotCI(document.getElementById("t-roi-chart"), { rows, xLabel: "node strength: summed absolute weight on the region's features", domain: [0, d3.max(rows, (r) => r.value) * 1.1] });
  const aal = d.interpret.weights.aal?.blocks;
  if (aal && state.atlas === "aal") {
    const key = state.blockKind;
    const names = aal.names;
    heatmap(document.getElementById("t-block-chart"), { rows: names, cols: names, values: aal[key], format: (v) => d3.format(key === "mean_abs" ? ".3f" : "+.3f")(v),
      scale: key === "mean_abs" ? "seq" : "div", label: key === "mean_abs" ? "mean |weight| per feature" : "mean weight per feature", cell: 26 });
  } else {
    pending(document.getElementById("t-block-chart"), "Lobe blocks are available for the AAL atlas, which names its regions.");
  }
}

function renderStability(d) {
  const st = d.interpret.stability;
  const rows = Object.entries(st).map(([a, s]) => ({ atlas: a, jac: s.top_k_jaccard_pairs, wrk: s.weight_rank_correlation, node: s.node_strength_rank_correlation, full: s.node_strength_vs_full }));
  renderTable(document.getElementById("t-stab-table"), [
    { key: "atlas", label: "Atlas" }, { key: "jac", label: "top-100 overlap", num: true, format: f2 },
    { key: "wrk", label: "weight rank corr.", num: true, format: f2 }, { key: "node", label: "node strength rank corr.", num: true, format: f2 },
    { key: "full", label: "node strength vs full fit", num: true, format: f2 },
  ], rows);
  const del = d.interpret.deletion?.aal;
  if (!del) return pending(document.getElementById("t-del-chart"));
  const fr = Object.keys(del.fractions).map(Number).sort((a, b) => a - b);
  const mk = (key) => fr.map((f) => ({ x: f, y: del.fractions[String(f)][key] }));
  lines(document.getElementById("t-del-chart"), { xLabel: "fraction of each subject's features deleted", yLabel: "held-out AUC", yDomain: [0.3, 0.8],
    xFormat: d3.format(".0%"), tipTitle: "deleted", series: [
      { label: "most supportive features", color: css("--series-2"), points: mk("supportive") },
      { label: "random features", color: css("--muted-series"), points: mk("random") },
      { label: "least supportive features", color: css("--series-3"), points: mk("least_supportive") },
    ] });
}

export async function initTrustFmri() {
  const d = await load();
  const ids = ["t-score", "t-cal-chart", "t-sel-chart", "t-short-chart", "t-group-chart", "t-roi-chart", "t-block-chart", "t-del-chart"];
  if (!d?.fmri) { ids.forEach((id) => pending(document.getElementById(id))); return; }
  const bind = (sel, key, redraw) => document.querySelectorAll(sel).forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(sel).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state[key] = b.dataset.v;
    redraw();
  }));
  const calSel = document.getElementById("t-cal-model");
  calSel.innerHTML = MODEL_ORDER.filter((m) => d.fmri.kfold.calibration[m]).map((m) => `<option value="${m}">${MODEL_NAME[m]}</option>`).join("");
  calSel.addEventListener("change", () => { state.calModel = calSel.value; renderCalibration(d); });
  bind("#t-cal-scheme button", "calScheme", () => renderCalibration(d));
  bind("#t-group button", "group", () => renderGroups(d));
  bind("#t-block-kind button", "blockKind", () => renderWeights(d));
  const atlasSel = document.getElementById("t-atlas");
  if (d.interpret) {
    atlasSel.innerHTML = Object.keys(d.interpret.weights).map((a) => `<option value="${a}">${a}</option>`).join("");
    atlasSel.value = state.atlas;
    atlasSel.addEventListener("change", () => { state.atlas = atlasSel.value; renderWeights(d); });
  }
  rerenderFmri = () => {
    scorecard(document.getElementById("t-score"), fmriChecks(d));
    renderCalibration(d); renderSelective(d); renderShortcuts(d); renderGroups(d);
    if (d.interpret) { renderWeights(d); renderStability(d); }
    else ["t-roi-chart", "t-block-chart", "t-del-chart"].forEach((id) => pending(document.getElementById(id)));
  };
  rerenderFmri();
}
let rerenderFmri = () => {};
export const redrawTrustFmri = () => rerenderFmri();

// ================================================================ segmentation

const REGIONS = ["WT", "TC", "ET"];
const REGION_NAME = { WT: "whole tumour", TC: "tumour core", ET: "enhancing tumour" };
const CONTRASTS = ["FLAIR", "T1", "T1Gd", "T2"];
const PERT = ["noise 0.2", "noise 0.5", "bias field", "blur", "no FLAIR", "no T1", "no T1Gd", "no T2"];

function runsFor(d, budget) {
  const r = d.smri?.runs ?? {};
  return { ours: r[`ours_${budget}`], base: r[`scratch_${budget}`], key: `ours_${budget}|scratch_${budget}` };
}

function smriChecks(d, budget) {
  const { ours, base, key } = runsFor(d, budget);
  if (!ours) return [];
  const out = [];
  const u = ours.uncertainty.WT.entropy.spearman;
  out.push({ name: "Uncertainty flags bad scans", status: u.rho >= 0.3 && u.ci_low > 0 ? "holds" : u.rho > 0 && u.ci_low > 0 ? "partly" : "fails",
    evidence: `Spearman correlation of entropy with error on whole tumour ${f2(u.rho)} [${f2(u.ci_low)}, ${f2(u.ci_high)}].`,
    rule: "ρ ≥ 0.3 with interval above 0 (partly: ρ > 0, interval above 0)" });
  const ece = REGIONS.map((r) => ours.calibration[r].ece);
  out.push({ name: "Probabilities mean what they say", status: grade(Math.max(...ece) <= 0.05, Math.max(...ece) <= 0.10),
    evidence: `Voxel ECE near the tumour: WT ${f3(ece[0])}, TC ${f3(ece[1])}, ET ${f3(ece[2])}.`, rule: "ECE ≤ 0.05 in all regions (partly: 0.10)" });
  const v = ours.volume.WT;
  out.push({ name: "Volumes can be followed", status: v.pearson >= 0.95 && v.relative_error <= 0.15 ? "holds" : v.pearson >= 0.9 ? "partly" : "fails",
    evidence: `Whole-tumour volume correlation ${f3(v.pearson)}, median relative error ${f2(v.relative_error)}, bias ${f2(v.bias)} mL.`,
    rule: "correlation ≥ 0.95 and relative error ≤ 0.15 (partly: correlation ≥ 0.90)" });
  const noise = ours.robustness.perturbations["noise 0.2"];
  out.push({ name: "Survives noisy images", status: grade(noise.drop_mean <= 0.02, noise.drop_mean <= 0.05),
    evidence: `Dice ${f3(ours.robustness.clean_mean)} → ${f3(noise.mean)} with noise of 0.2 standard deviations.`, rule: "drop ≤ 0.02 (partly: 0.05)" });
  const drop = (c, r) => ours.robustness.perturbations[`no ${c}`]?.drop[r] ?? -1;
  const flairOk = CONTRASTS.every((c) => c === "FLAIR" || drop("FLAIR", "WT") > drop(c, "WT"));
  const gdOk = CONTRASTS.every((c) => c === "T1Gd" || drop("T1Gd", "ET") > drop(c, "ET"));
  out.push({ name: "Relies on the contrasts radiologists use", status: flairOk && gdOk ? "holds" : flairOk || gdOk ? "partly" : "fails",
    evidence: `Without FLAIR the whole tumour loses ${f2(drop("FLAIR", "WT"))} Dice; without T1Gd the enhancing tumour loses ${f2(drop("T1Gd", "ET"))}.`,
    rule: "FLAIR matters most for WT and T1Gd matters most for ET" });
  const blind = ours.robustness.perturbations["mirror blind"];
  if (blind) {
    out.push({ name: "The mirror channel is used", status: grade(blind.drop_mean >= 0.01, blind.drop_mean > 0),
      evidence: `Dice ${f3(ours.robustness.clean_mean)} → ${f3(blind.mean)} when the blocks cannot see the mirror image.`, rule: "drop ≥ 0.01 (partly: > 0)" });
  }
  const cmp = d.smri.robustness_comparisons?.[key];
  if (cmp) {
    const better = Object.values(cmp).filter((c) => c.mean_diff > 0).length, total = Object.keys(cmp).length;
    out.push({ name: "More robust than the baseline", status: grade(better >= 6, better >= 4),
      evidence: `Ours loses less Dice than the U-Net from scratch under ${better} of ${total} perturbations.`, rule: "≥ 6 of 8 (partly: 4)" });
  }
  return out;
}

function renderUncertainty(d) {
  const { ours } = runsFor(d, state.smriBudget);
  const r = state.smriRegion;
  const rows = ours.points?.[r];
  if (!rows) { pending(document.getElementById("t-unc-chart")); return; }
  scatter(document.getElementById("t-unc-chart"), { points: rows.map((p) => ({ id: p.id, x: p.u, y: 1 - p.dice })),
    xLabel: "mean entropy over the plausible region", yLabel: "error, 1 − Dice", color: css("--series-1"),
    note: `Spearman ρ = ${f2(ours.uncertainty[r].entropy.spearman.rho)}` });
  const rc = ours.uncertainty[r].entropy.risk_coverage;
  lines(document.getElementById("t-rc-chart"), { xLabel: "fraction of scans kept (least uncertain first)", yLabel: "mean error of the kept scans",
    xDomain: [0, 1], xFormat: d3.format(".0%"), tipTitle: "kept", series: [
      { label: "ranked by entropy", color: css("--series-1"), points: rc.coverage.map((c, i) => ({ x: c, y: rc.risk[i] })) },
      { label: "ideal ranking", color: css("--axis"), dashed: true, points: rc.coverage.map((c, i) => ({ x: c, y: rc.oracle_risk[i] })) },
      { label: "random order", color: css("--muted-series"), dashed: true, points: rc.coverage.map((c) => ({ x: c, y: rc.random_aurc })) },
    ] });
}

function renderSegCal(d) {
  const { ours, base } = runsFor(d, state.smriBudget);
  const r = state.smriRegion;
  const mk = (run) => run.calibration[r].bins.map((b) => ({ x: b.mean_pred, y: b.observed, n: b.n }));
  const series = [{ label: `ours (ECE ${f3(ours.calibration[r].ece)})`, color: css("--series-1"), points: mk(ours) }];
  if (base) series.push({ label: `U-Net from scratch (ECE ${f3(base.calibration[r].ece)})`, color: css("--series-2"), points: mk(base) });
  reliability(document.getElementById("t-segcal-chart"), { series, xLabel: `predicted probability, ${REGION_NAME[r]}`,
    yLabel: "observed fraction of voxels" });
}

function renderVolumes(d) {
  const { ours } = runsFor(d, state.smriBudget);
  const r = state.smriRegion, vp = ours.volume_points[r], a = ours.volume[r];
  const pts = vp.case.map((id, i) => ({ id, x: vp.true[i], y: vp.pred[i], xs: `${vp.true[i].toFixed(1)} mL`, ys: `${vp.pred[i].toFixed(1)} mL` })).filter((p) => p.x > 0);
  scatter(document.getElementById("t-vol-chart"), { points: pts, xLabel: "reference volume, mL", yLabel: "predicted volume, mL", color: css("--series-1"),
    log: true, identity: true, note: `r = ${f3(a.pearson)}` });
  document.getElementById("t-vol-note").textContent =
    `${REGION_NAME[r]}: bias ${f2(a.bias)} mL, 95% limits of agreement ${f2(a.loa_low)} to ${f2(a.loa_high)} mL, median relative error ${f2(a.relative_error)}.`;
}

function renderFailures(d) {
  const { ours, base } = runsFor(d, state.smriBudget);
  const sizes = Object.keys(ours.by_size);
  const rows = sizes.map((s) => ({ label: `${s} · ${Math.round(ours.by_size[s].wt_range_ml[0])}–${Math.round(ours.by_size[s].wt_range_ml[1])} mL`, n: ours.by_size[s].cases,
    a: ours.by_size[s][`dice_${state.smriRegion}`], b: base ? base.by_size[s][`dice_${state.smriRegion}`] : ours.by_size[s][`dice_${state.smriRegion}`] }));
  dumbbell(document.getElementById("t-size-chart"), { rows, aLabel: "ours", bLabel: "U-Net from scratch", aColor: css("--series-1"), bColor: css("--series-2"),
    xLabel: `Dice, ${REGION_NAME[state.smriRegion]}`, tip: (r) => [r.label, [["scans", r.n], ["ours", f3(r.a)], ["U-Net from scratch", f3(r.b)]]] });
  renderTable(document.getElementById("t-worst"), [
    { key: "case", label: "Scan" }, { key: "dice_mean", label: "mean Dice", num: true, format: f3 }, { key: "why", label: "What went wrong" },
  ], ours.worst_cases.slice(0, 6).map((w) => ({ case: w.case, dice_mean: w.dice_mean, why: w.tags.join("; ") })));
  const et = ours.et_absent;
  document.getElementById("t-et-note").textContent = `Enhancing tumour is absent in ${et.cases} test scans; the model predicted some anyway in ${et.false_alarm} of them.`;
}

function renderRobust(d) {
  const { ours, base, key } = runsFor(d, state.smriBudget);
  const names = ["clean", ...PERT.filter((p) => ours.robustness.perturbations[p])];
  const val = (run, n) => (n === "clean" ? run.robustness.clean_mean : run.robustness.perturbations[n]?.mean);
  const rows = names.map((n) => ({ label: n, a: val(ours, n), b: base ? val(base, n) : val(ours, n), n: ours.robustness.scans }));
  dumbbell(document.getElementById("t-rob-chart"), { rows, aLabel: "ours", bLabel: "U-Net from scratch", aColor: css("--series-1"), bColor: css("--series-2"),
    xLabel: "mean Dice over WT, TC, ET", tip: (r) => [r.label, [["ours", f3(r.a)], ["U-Net from scratch", f3(r.b)], ["scans", r.n]]] });
  const cmp = d.smri.robustness_comparisons?.[key];
  const el = document.getElementById("t-rob-table");
  if (!cmp) { el.innerHTML = ""; return; }
  renderTable(el, [
    { key: "name", label: "Perturbation" }, { key: "diff", label: "ours loses less by", num: true, format: (v) => d3.format("+.3f")(v) },
    { key: "ci", label: "95% interval" }, { key: "wl", label: "scans better / worse", num: true }, { key: "p", label: "p", num: true, format: (v) => (v < 0.001 ? "<0.001" : f3(v)) },
  ], Object.entries(cmp).map(([name, c]) => ({ name, diff: c.mean_diff, ci: `${d3.format("+.3f")(c.ci_low)} to ${d3.format("+.3f")(c.ci_high)}`, wl: `${c.wins}/${c.losses}`, p: c.wilcoxon_p_one_sided })));
}

function renderReliance(d) {
  const { ours, base } = runsFor(d, state.smriBudget);
  const matrix = (run) => CONTRASTS.map((c) => REGIONS.map((r) => run.robustness.perturbations[`no ${c}`]?.drop[r] ?? null));
  heatmap(document.getElementById("t-rel-ours"), { rows: CONTRASTS.map((c) => `without ${c}`), cols: REGIONS.map((r) => REGION_NAME[r]), values: matrix(ours),
    format: f2, label: "Dice lost", cell: 34 });
  if (base) heatmap(document.getElementById("t-rel-base"), { rows: CONTRASTS.map((c) => `without ${c}`), cols: REGIONS.map((r) => REGION_NAME[r]), values: matrix(base),
    format: f2, label: "Dice lost", cell: 34 });
  const sal = ours.saliency_share;
  if (sal) {
    const rows = CONTRASTS.map((c) => ({ label: c, value: sal[c], color: css("--series-1") }));
    dotCI(document.getElementById("t-sal-chart"), { rows, xLabel: "share of attribution inside the segmented tumour", domain: [0, 0.6] });
  } else pending(document.getElementById("t-sal-chart"));
}

function renderBilateral(d) {
  const { ours } = runsFor(d, state.smriBudget);
  const ex = ours.exchange;
  const box = document.getElementById("t-bil-card");
  if (!ex?.length) { box.hidden = true; return; }
  box.hidden = false;
  const lv = ex.map((e, i) => ({ e, level: i + 1 })).filter((x) => x.e && x.e.scans);
  const rows = lv.map((x) => ({ label: `level ${x.level}`, a: x.e.correction_in, b: x.e.correction_out, n: x.e.scans }));
  dumbbell(document.getElementById("t-bil-chart"), { rows, aLabel: "inside the tumour", bLabel: "outside the tumour", aColor: css("--series-2"), bColor: css("--series-3"),
    xLabel: "size of the block's update relative to its input", tip: (r) => [r.label, [["inside tumour", f3(r.a)], ["outside", f3(r.b)], ["scans", r.n]]] });
  const blind = ours.robustness.perturbations["mirror blind"];
  document.getElementById("t-bil-note").innerHTML =
    `Asymmetry signal |f − Rf| alone finds the tumour with AUC ${lv.map((x) => `${f2(x.e.asymmetry_auc)} (level ${x.level})`).join(", ")}. ` +
    (blind ? `With the mirror image replaced by the scan's own features, mean Dice goes from ${f3(ours.robustness.clean_mean)} to ${f3(blind.mean)}.` : "");
}

export async function initTrustSmri() {
  const d = await load();
  const ids = ["t-score-s", "t-unc-chart", "t-rc-chart", "t-segcal-chart", "t-vol-chart", "t-size-chart", "t-rob-chart", "t-rel-ours", "t-sal-chart"];
  if (!d?.smri?.runs || !Object.keys(d.smri.runs).length) { ids.forEach((id) => pending(document.getElementById(id))); return; }
  const budgets = [...new Set(Object.keys(d.smri.runs).filter((k) => k.startsWith("ours_")).map((k) => k.slice(5)))];
  const sel = document.getElementById("t-budget");
  sel.innerHTML = budgets.map((b) => `<option value="${b}">${b.replace("pct", "% of labels")}</option>`).join("");
  state.smriBudget = budgets[0];
  sel.addEventListener("change", () => { state.smriBudget = sel.value; redraw(); });
  document.querySelectorAll("#t-region button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#t-region button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.smriRegion = b.dataset.v;
    redraw();
  }));
  const redraw = () => {
    scorecard(document.getElementById("t-score-s"), smriChecks(d, state.smriBudget));
    renderUncertainty(d); renderSegCal(d); renderVolumes(d); renderFailures(d); renderRobust(d); renderReliance(d); renderBilateral(d);
  };
  rerenderSmri = redraw;
  redraw();
}
let rerenderSmri = () => {};
export const redrawTrustSmri = () => rerenderSmri();
