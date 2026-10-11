// Interactive explainers for the method page, and the pipeline diagram.

import { css, d3, hideTip, showTip } from "./charts.js";

// ---------------------------------------------------------------- bilateral exchange

// A synthetic axial FLAIR-like slice: two hemispheres with gyri, mirrored
// ventricles, and a tumour with an oedema halo in the right hemisphere.
function syntheticSlice(w, h, cx, cy, rx, ry, tumour) {
  const field = (x, y) => {
    const dx = (x - cx) / rx, dy = (y - cy) / ry;
    const r = Math.hypot(dx, dy * 0.96);
    if (r > 1) return null;
    const ang = Math.atan2(dy, Math.abs(dx));
    let v = 0.46 + 0.1 * Math.cos(5 * dy);                               // white matter
    const rim = Math.max(0, (r - 0.72) / 0.28);
    v += rim * (0.16 * Math.cos(22 * ang + 9 * r) + 0.1);               // cortical ribbon and gyri
    const fiss = Math.exp(-((x - cx) ** 2) / 6) * Math.max(0, dy + 0.15);
    v -= 0.55 * fiss;                                                    // interhemispheric fissure
    for (const s of [-1, 1]) {                                           // ventricles, mirror symmetric
      const vx = (x - (cx + s * 0.16 * rx)) / (0.09 * rx), vy = (y - (cy - 0.05 * ry)) / (0.32 * ry);
      if (vx * vx + vy * vy < 1) v -= 0.32 * (1 - vx * vx - vy * vy);
    }
    const d = Math.hypot(x - tumour.x, y - tumour.y);
    v += 0.42 * Math.exp(-(d * d) / (2 * (tumour.r * 1.7) ** 2));       // oedema halo
    v += 0.35 * Math.exp(-(d * d) / (2 * (tumour.r * 0.75) ** 2));
    v -= 0.3 * Math.exp(-(d * d) / (2 * (tumour.r * 0.32) ** 2));       // necrotic centre
    v *= 1 - Math.max(0, (r - 0.97) / 0.03);
    return Math.max(0, Math.min(1, v));
  };
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = field(x, y);
    const k = 4 * (y * w + x);
    const g = v == null ? 0 : Math.round(18 + 225 * v);
    img.data[k] = g; img.data[k + 1] = g; img.data[k + 2] = Math.min(255, g + 6); img.data[k + 3] = v == null ? 0 : 255;
  }
  ctx.putImageData(img, 0, 0);
  return { url: canvas.toDataURL(), field };
}

function bilateral(el) {
  el.innerHTML = "";
  const w = 360, h = 300, cx = w / 2, cy = 132, rx = 108, ry = 118;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const tumour = { x: cx + 50, y: cy - 30, r: 15 };
  const slice = syntheticSlice(w, h, cx, cy, rx, ry, tumour);
  const field = (x, y) => {
    let s = 0, n = 0;                                                    // local mean, a stand-in for a feature
    for (let dx = -4; dx <= 4; dx += 2) for (let dy = -4; dy <= 4; dy += 2) {
      const v = slice.field(x + dx, y + dy);
      if (v != null) { s += v; n++; }
    }
    return n ? s / n : 0;
  };
  svg.append("image").attr("href", slice.url).attr("width", w).attr("height", h);
  svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", cy - ry - 4).attr("y2", cy + ry + 4)
    .attr("stroke", css("--text-muted")).attr("stroke-width", 1).attr("opacity", 0.6);
  svg.append("text").attr("x", cx + 5).attr("y", 12).attr("fill", css("--text-muted")).style("font-size", "11px").text("midline");
  const probe = svg.append("circle").attr("r", 7).attr("fill", "none").attr("stroke", css("--series-1")).attr("stroke-width", 2.5).style("cursor", "grab");
  const mirror = svg.append("circle").attr("r", 7).attr("fill", "none").attr("stroke", css("--series-2")).attr("stroke-width", 2.5);
  const link = svg.append("line").attr("stroke", css("--text-muted")).attr("stroke-width", 1);
  const readout = svg.append("g").attr("transform", `translate(20,${h - 36})`);
  const bars = [["f(x)", css("--series-1")], ["f(Rx)", css("--series-2")], ["|f - Rf|", css("--text-primary")]];
  bars.forEach(([label], i) => readout.append("text").attr("x", i * 112).attr("y", 0).attr("fill", css("--text-secondary")).style("font-size", "12px").text(label));
  const rects = bars.map(([, color], i) => readout.append("rect").attr("x", i * 112).attr("y", 8).attr("height", 10).attr("rx", 3).attr("fill", color));
  const place = (x, y) => {
    const mx = 2 * cx - x;
    probe.attr("cx", x).attr("cy", y);
    mirror.attr("cx", mx).attr("cy", y);
    link.attr("x1", x).attr("x2", mx).attr("y1", y).attr("y2", y);
    const a = field(x, y), b = field(mx, y);
    rects[0].attr("width", 96 * a);
    rects[1].attr("width", 96 * b);
    rects[2].attr("width", 96 * Math.abs(a - b) * 1.4);
  };
  place(tumour.x, tumour.y);
  svg.call(d3.drag().on("drag", (event) => {
    const x = Math.max(cx - rx + 8, Math.min(cx + rx - 8, event.x)), y = Math.max(cy - ry + 8, Math.min(cy + ry - 8, event.y));
    if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 0.95) place(x, y);
  }));
}

// ---------------------------------------------------------------- masks

function masks(el) {
  el.innerHTML = "";
  const nx = 10, ny = 8, cell = 15, gap = 2, pad = 26;
  const w = 2 * (nx * cell) + 3 * pad, h = ny * cell + 112;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const draw = () => {
    svg.selectAll("*").remove();
    const random = Array.from({ length: nx * ny }, () => Math.random() < 0.6);
    const contra = new Array(nx * ny).fill(false);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx / 2; i++) {
      const both = Math.random() < 0.2, left = Math.random() < 0.5;
      contra[j * nx + i] = both || left;
      contra[j * nx + (nx - 1 - i)] = both || !left;
    }
    [["random masking", random, pad], ["contralateral masking", contra, 2 * pad + nx * cell]].forEach(([title, mask, x0]) => {
      const g = svg.append("g").attr("transform", `translate(${x0},26)`);
      g.append("text").attr("y", -10).attr("fill", css("--text-secondary")).style("font-size", "12px").text(title);
      g.append("line").attr("x1", (nx * cell) / 2).attr("x2", (nx * cell) / 2).attr("y1", -2).attr("y2", ny * cell + 2).attr("stroke", css("--text-muted")).attr("stroke-dasharray", "2 3");
      const rect = g.selectAll("rect").data(mask.map((m, k) => ({ m, i: k % nx, j: Math.floor(k / nx) }))).join("rect")
        .attr("x", (d) => d.i * cell + gap / 2).attr("y", (d) => d.j * cell + gap / 2).attr("width", cell - gap).attr("height", cell - gap).attr("rx", 2)
        .attr("fill", (d) => (d.m ? css("--surface-3") : css("--series-1"))).attr("fill-opacity", (d) => (d.m ? 1 : 0.75));
      rect.on("mouseenter", function (event, d) {
        rect.attr("stroke", (e) => (e.j === d.j && (e.i === d.i || e.i === nx - 1 - d.i) ? css("--text-primary") : "none")).attr("stroke-width", 1.5);
        const twin = mask[d.j * nx + (nx - 1 - d.i)];
        showTip(event, d.m ? "hidden cube" : "visible cube", [["mirror twin", twin ? "hidden" : "visible"]]);
      }).on("mouseleave", () => { rect.attr("stroke", "none"); hideTip(); });
      const hiddenWithVisibleTwin = mask.filter((m, k) => m && !mask[Math.floor(k / nx) * nx + (nx - 1 - (k % nx))]).length;
      const hidden = mask.filter(Boolean).length;
      g.append("text").attr("y", ny * cell + 20).attr("fill", css("--text-primary")).style("font-size", "15px").style("font-weight", 600)
        .text(`${Math.round((100 * hiddenWithVisibleTwin) / hidden)}%`);
      g.append("text").attr("y", ny * cell + 36).attr("fill", css("--text-muted")).style("font-size", "11px")
        .text("of hidden cubes have");
      g.append("text").attr("y", ny * cell + 50).attr("fill", css("--text-muted")).style("font-size", "11px")
        .text("a visible mirror twin");
    });
    const key = svg.append("g").attr("transform", `translate(${pad},${h - 14})`);
    key.append("rect").attr("width", 10).attr("height", 10).attr("y", -9).attr("rx", 2).attr("fill", css("--series-1")).attr("fill-opacity", 0.75);
    key.append("text").attr("x", 15).attr("fill", css("--text-secondary")).style("font-size", "11.5px").text("visible");
    key.append("rect").attr("x", 70).attr("width", 10).attr("height", 10).attr("y", -9).attr("rx", 2).attr("fill", css("--surface-3"));
    key.append("text").attr("x", 85).attr("fill", css("--text-secondary")).style("font-size", "11.5px").text("hidden");
    svg.append("text").attr("x", w - pad).attr("y", h - 5).attr("text-anchor", "end").attr("fill", css("--series-1"))
      .style("font-size", "12px").style("cursor", "pointer").text("draw new masks").on("click", draw);
  };
  draw();
}

// ---------------------------------------------------------------- vicinal training

function logistic(points, iters = 600, lam = 0.02) {
  let w0 = 0, w1 = 0, b = 0;
  for (let t = 0; t < iters; t++) {
    let g0 = lam * w0, g1 = lam * w1, gb = 0, tot = 0;
    for (const p of points) {
      const z = 1 / (1 + Math.exp(-(w0 * p.x + w1 * p.y + b))) - p.label;
      g0 += p.wt * z * p.x; g1 += p.wt * z * p.y; gb += p.wt * z; tot += p.wt;
    }
    w0 -= (0.8 * g0) / tot; w1 -= (0.8 * g1) / tot; b -= (0.8 * gb) / tot;
  }
  return { w0, w1, b };
}

function vicinal(el) {
  el.innerHTML = "";
  const w = 360, h = 300, m = 16;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const rand = d3.randomNormal.source(d3.randomLcg(7));
  const n0 = rand(0, 1);
  // Two features: x carries the diagnosis and is reliable; y is noisy within a scan
  // and happens to separate the training subjects a little.
  const subjects = [];
  for (let i = 0; i < 26; i++) {
    const label = i % 2;
    const cxv = (label ? 0.55 : -0.55) + 0.55 * n0();
    const cyv = (label ? 0.35 : -0.35) + 0.45 * n0();
    const windows = Array.from({ length: 6 }, () => ({ x: cxv + 0.1 * n0(), y: cyv + 0.75 * n0() }));
    subjects.push({ label, x: cxv, y: cyv, windows });
  }
  const x = d3.scaleLinear().domain([-2.2, 2.2]).range([m, w - m]);
  const y = d3.scaleLinear().domain([-2.2, 2.2]).range([h - m - 30, m]);
  const erm = logistic(subjects.map((s) => ({ ...s, wt: 1 })));
  const vic = logistic(subjects.flatMap((s) => [{ ...s, wt: 1 }, ...s.windows.map((p) => ({ ...p, label: s.label, wt: 1 / 6 }))]));
  const boundary = (p, color, dash) => {
    const xs = [-2.2, 2.2];
    const ys = xs.map((v) => -(p.w0 * v + p.b) / p.w1);
    svg.append("line").attr("x1", x(xs[0])).attr("x2", x(xs[1])).attr("y1", y(ys[0])).attr("y2", y(ys[1]))
      .attr("stroke", color).attr("stroke-width", 2).attr("stroke-dasharray", dash);
  };
  const clouds = svg.append("g").attr("opacity", 0);
  subjects.forEach((s) => s.windows.forEach((p) => clouds.append("line").attr("x1", x(s.x)).attr("y1", y(s.y)).attr("x2", x(p.x)).attr("y2", y(p.y))
    .attr("stroke", s.label ? css("--series-5") : css("--series-3")).attr("stroke-opacity", 0.4)));
  svg.append("g").selectAll("circle").data(subjects).join("circle").attr("cx", (s) => x(s.x)).attr("cy", (s) => y(s.y)).attr("r", 4.5)
    .attr("fill", (s) => (s.label ? css("--series-5") : css("--series-3"))).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5);
  svg.append("defs").append("clipPath").attr("id", "vic-clip").append("rect").attr("x", m).attr("y", m).attr("width", w - 2 * m).attr("height", h - 2 * m - 30);
  const lines = svg.append("g").attr("clip-path", "url(#vic-clip)");
  const b1 = { ...erm }, b2 = { ...vic };
  [[b1, css("--series-2"), "5 4"], [b2, css("--series-1"), null]].forEach(([p, c, dash]) => {
    const xs = [-2.2, 2.2], ys = xs.map((v) => -(p.w0 * v + p.b) / p.w1);
    lines.append("line").attr("x1", x(xs[0])).attr("x2", x(xs[1])).attr("y1", y(ys[0])).attr("y2", y(ys[1])).attr("stroke", c).attr("stroke-width", 2).attr("stroke-dasharray", dash);
  });
  const foot = svg.append("g").attr("transform", `translate(${m},${h - 14})`);
  foot.append("line").attr("x1", 0).attr("x2", 18).attr("y1", -4).attr("y2", -4).attr("stroke", css("--series-2")).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
  foot.append("text").attr("x", 24).attr("y", 0).attr("fill", css("--text-secondary")).style("font-size", "11.5px").text("one point per subject");
  foot.append("line").attr("x1", 160).attr("x2", 178).attr("y1", -4).attr("y2", -4).attr("stroke", css("--series-1")).attr("stroke-width", 2);
  foot.append("text").attr("x", 184).attr("y", 0).attr("fill", css("--text-secondary")).style("font-size", "11.5px").text("with window clouds");
  const classes = svg.append("g").attr("transform", `translate(${m},${m + 4})`);
  [["control", css("--series-3")], ["autism", css("--series-5")]].forEach(([label, color], i) => {
    classes.append("circle").attr("cx", 5 + i * 70).attr("cy", -4).attr("r", 4.5).attr("fill", color);
    classes.append("text").attr("x", 13 + i * 70).attr("fill", css("--text-secondary")).style("font-size", "11.5px").text(label);
  });
  const toggle = svg.append("text").attr("x", w - m).attr("y", m + 4).attr("text-anchor", "end").attr("fill", css("--series-1"))
    .style("font-size", "12px").style("cursor", "pointer").text("show window clouds");
  let on = false;
  toggle.on("click", () => { on = !on; clouds.transition().duration(300).attr("opacity", on ? 1 : 0); toggle.text(on ? "hide window clouds" : "show window clouds"); });
  void boundary;
}

export function initExplainers() {
  bilateral(document.getElementById("ex-bilateral"));
  masks(document.getElementById("ex-mask"));
  vicinal(document.getElementById("ex-vicinal"));
}

// ---------------------------------------------------------------- pipeline diagram

export function pipelineDag(el) {
  el.innerHTML = "";
  const nodes = [
    { id: "download", label: "DOWNLOAD", x: 0, y: 1 },
    { id: "pmsd", label: "PREPARE_MSD", x: 1, y: 0 }, { id: "pab", label: "PREPARE_ABIDE", x: 1, y: 2 },
    { id: "pre", label: "SMRI_PRETRAIN", x: 2, y: 0 }, { id: "cv", label: "FMRI_CV", x: 2.5, y: 2 },
    { id: "ft", label: "SMRI_FINETUNE", x: 3, y: 0 }, { id: "ev", label: "SMRI_EVALUATE", x: 4, y: 0 },
    { id: "export", label: "EXPORT_MODELS", x: 5, y: 0.6 }, { id: "report", label: "REPORT", x: 5, y: 1.6 },
  ];
  const edges = [["download", "pmsd"], ["download", "pab"], ["pmsd", "pre"], ["pre", "ft"], ["pmsd", "ft"], ["ft", "ev"],
    ["pab", "cv"], ["ev", "report"], ["cv", "report"], ["ft", "export"], ["cv", "export"]];
  const W = Math.max(560, el.clientWidth || 700), H = 300, nw = 118, nh = 34;
  const x = d3.scaleLinear().domain([0, 5]).range([nw / 2 + 4, W - nw / 2 - 4]);
  const y = d3.scaleLinear().domain([0, 2]).range([40, H - 40]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const pos = Object.fromEntries(nodes.map((n) => [n.id, [x(n.x), y(n.y)]]));
  svg.append("g").selectAll("path").data(edges).join("path")
    .attr("d", ([a, b]) => {
      const [x1, y1] = pos[a], [x2, y2] = pos[b];
      const sx = x1 + nw / 2, tx = x2 - nw / 2, mx = (sx + tx) / 2;
      return `M${sx},${y1} C${mx},${y1} ${mx},${y2} ${tx},${y2}`;
    }).attr("fill", "none").attr("stroke", css("--axis")).attr("stroke-width", 1.5);
  const g = svg.append("g").selectAll("g").data(nodes).join("g").attr("transform", (n) => `translate(${pos[n.id][0] - nw / 2},${pos[n.id][1] - nh / 2})`);
  g.append("rect").attr("width", nw).attr("height", nh).attr("rx", 8).attr("fill", css("--surface-3")).attr("stroke", css("--border-strong"));
  g.append("text").attr("x", nw / 2).attr("y", nh / 2 + 4).attr("text-anchor", "middle").attr("fill", css("--text-primary"))
    .style("font", `500 11px ${css("--font-mono")}`).text((n) => n.label);
  const lane = (label, yy) => svg.append("text").attr("x", 4).attr("y", yy).attr("fill", css("--text-muted")).style("font-size", "11px").text(label);
  lane("structural MRI", y(0) - nh / 2 - 8);
  lane("resting-state fMRI", y(2) + nh / 2 + 18);
}
