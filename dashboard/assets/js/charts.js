// Chart components drawn with D3 into inline SVG.
// Colours and type come from CSS custom properties, so a theme switch only
// needs a re-render. Every chart registers a table twin for the "Table" toggle.

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const tooltip = () => document.getElementById("tooltip");
export function showTip(event, title, rows) {
  const tip = tooltip();
  tip.innerHTML = `<div class="t-title">${title}</div>` +
    rows.map(([k, v]) => `<div class="t-row"><span>${k}</span><b>${v}</b></div>`).join("");
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + rect.width > w - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > h - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
  tip.classList.add("show");
}
export function hideTip() { tooltip().classList.remove("show"); }

const fmt3 = d3.format(".3f");
export const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : d3.format(`.${d}f`)(v));

function frame(el, { width, height, margin }) {
  el.innerHTML = "";
  const w = width ?? Math.max(320, el.clientWidth || 600);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${height}`).attr("role", "img");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { svg, g, w, iw: w - margin.left - margin.right, ih: height - margin.top - margin.bottom };
}

function legend(el, items) {
  const div = document.createElement("div");
  div.className = "legend";
  div.style.marginBottom = "8px";
  div.innerHTML = items.map((i) => `<span><i class="swatch" style="background:${i.color}"></i>${i.label}</span>`).join("");
  el.prepend(div);
}

// ---------------------------------------------------------------- table twins

const tables = new Map();
export function registerTable(chartId, columns, rows) { tables.set(chartId, { columns, rows }); }

export function renderTable(el, columns, rows) {
  const head = columns.map((c) => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("");
  const body = rows.map((r) => `<tr class="${r._ours ? "ours" : ""}">` + columns.map((c) =>
    `<td class="${c.num ? "num" : ""}">${c.format ? c.format(r[c.key]) : r[c.key] ?? "–"}</td>`).join("") + "</tr>").join("");
  el.innerHTML = `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

export function wireTableToggles(renderers) {
  document.querySelectorAll("[data-table-for]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.tableFor;
      const el = document.getElementById(id);
      const showing = btn.getAttribute("aria-pressed") === "true";
      btn.setAttribute("aria-pressed", String(!showing));
      btn.textContent = showing ? "Table" : "Chart";
      if (!showing && tables.has(id)) {
        const { columns, rows } = tables.get(id);
        renderTable(el, columns, rows);
      } else if (renderers[id]) {
        renderers[id]();
      }
    });
  });
}

// ---------------------------------------------------------------- line with interval bars

export function lineCI(el, { series, xLabel, yLabel, xTicks, yDomain, height = 330 }) {
  const margin = { top: 14, right: 96, bottom: 42, left: 52 };
  const { g, iw, ih } = frame(el, { height, margin });
  const xs = xTicks;
  const x = d3.scalePoint().domain(xs).range([0, iw]).padding(0.3);
  const all = series.flatMap((s) => s.points.flatMap((p) => [p.lo ?? p.y, p.hi ?? p.y]));
  const y = d3.scaleLinear().domain(yDomain ?? [d3.min(all) - 0.03, Math.min(1, d3.max(all) + 0.02)]).nice().range([ih, 0]);

  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).tickSizeOuter(0))
    .call((a) => a.select(".domain").attr("stroke", css("--axis")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 36).attr("text-anchor", "middle").text(xLabel);
  g.append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -40).attr("text-anchor", "middle").text(yLabel);

  const offset = d3.scalePoint().domain(series.map((s) => s.key)).range([-7, 7]);
  series.forEach((s) => {
    const color = s.color;
    const pts = s.points.filter((p) => p.y != null);
    const line = d3.line().x((p) => x(p.x) + offset(s.key)).y((p) => y(p.y));
    g.append("path").datum(pts).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2)
      .attr("stroke-linejoin", "round").attr("stroke-linecap", "round").attr("d", line);
    g.append("g").selectAll("line").data(pts.filter((p) => p.lo != null)).join("line")
      .attr("x1", (p) => x(p.x) + offset(s.key)).attr("x2", (p) => x(p.x) + offset(s.key))
      .attr("y1", (p) => y(p.lo)).attr("y2", (p) => y(p.hi)).attr("stroke", color).attr("stroke-width", 2).attr("stroke-linecap", "round").attr("opacity", 0.55);
    g.append("g").selectAll("circle").data(pts).join("circle")
      .attr("cx", (p) => x(p.x) + offset(s.key)).attr("cy", (p) => y(p.y)).attr("r", 4.5)
      .attr("fill", color).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
    const last = pts[pts.length - 1];
    if (last) {
      g.append("text").attr("class", s.emphasis ? "label-strong" : "label")
        .attr("x", x(last.x) + offset(s.key) + 12).attr("y", y(last.y) + 4).text(s.label);
    }
  });

  // hover: a column per x value
  const band = iw / xs.length;
  g.append("g").selectAll("rect").data(xs).join("rect")
    .attr("x", (d) => x(d) - band / 2).attr("width", band).attr("y", 0).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (event, d) => {
      const rows = series.map((s) => {
        const p = s.points.find((q) => q.x === d);
        return [s.label, p ? `${fmt3(p.y)}${p.lo != null ? ` [${fmt3(p.lo)}, ${fmt3(p.hi)}]` : ""}` : "–"];
      });
      showTip(event, `${d} of labels`, rows);
    })
    .on("mouseleave", hideTip);
  legend(el, series.map((s) => ({ label: s.label, color: s.color })));
}

// ---------------------------------------------------------------- dot with interval (rankings)

export function dotCI(el, { rows, xLabel, domain, height, valueFormat = fmt3, reference }) {
  const rowH = 30;
  const margin = { top: 8, right: 70, bottom: 40, left: Math.min(230, Math.max(...rows.map((r) => r.label.length)) * 7 + 18) };
  const h = height ?? margin.top + margin.bottom + rows.length * rowH;
  const { g, iw, ih } = frame(el, { height: h, margin });
  const lo = d3.min(rows, (r) => r.lo ?? r.value), hi = d3.max(rows, (r) => r.hi ?? r.value);
  const x = d3.scaleLinear().domain(domain ?? [lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.1]).nice().range([0, iw]);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, ih]).padding(0.35);
  g.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line")
    .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", 0).attr("y2", ih);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(10)).call((a) => a.select(".domain").remove())
    .selectAll("text").attr("class", (d) => rows.find((r) => r.label === d)?.emphasis ? "label-strong" : "label");
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle").text(xLabel);
  if (reference != null) {
    g.append("line").attr("x1", x(reference)).attr("x2", x(reference)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", css("--axis")).attr("stroke-width", 1);
  }
  const cy = (r) => y(r.label) + y.bandwidth() / 2;
  g.append("g").selectAll("line").data(rows.filter((r) => r.lo != null)).join("line")
    .attr("x1", (r) => x(r.lo)).attr("x2", (r) => x(r.hi)).attr("y1", cy).attr("y2", cy)
    .attr("stroke", (r) => r.color).attr("stroke-width", 2).attr("stroke-linecap", "round").attr("opacity", 0.6);
  g.append("g").selectAll("circle").data(rows).join("circle")
    .attr("cx", (r) => x(r.value)).attr("cy", cy).attr("r", (r) => (r.emphasis ? 6 : 4.5))
    .attr("fill", (r) => r.color).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
  g.append("g").selectAll("text").data(rows).join("text")
    .attr("class", (r) => (r.emphasis ? "label-strong" : "label")).attr("x", iw + 8).attr("y", (r) => cy(r) + 4).text((r) => valueFormat(r.value));
  g.append("g").selectAll("rect").data(rows).join("rect")
    .attr("x", -margin.left).attr("width", iw + margin.left + margin.right).attr("y", (r) => y(r.label) - 6).attr("height", y.bandwidth() + 12)
    .attr("fill", "transparent")
    .on("mousemove", (event, r) => showTip(event, r.label, [["value", valueFormat(r.value)], ...(r.lo != null ? [["95% interval", `${valueFormat(r.lo)} to ${valueFormat(r.hi)}`]] : []), ...(r.extra ?? [])]))
    .on("mouseleave", hideTip);
}

// ---------------------------------------------------------------- paired scatter

export function pairedScatter(el, { points, xLabel, yLabel, color, height = 330 }) {
  const margin = { top: 12, right: 18, bottom: 44, left: 52 };
  const { g, iw, ih } = frame(el, { height, margin });
  const ext = d3.extent(points.flatMap((p) => [p.x, p.y]));
  const lo = Math.max(0, Math.floor((ext[0] - 0.02) * 10) / 10);
  const x = d3.scaleLinear().domain([lo, 1]).range([0, iw]);
  const y = d3.scaleLinear().domain([lo, 1]).range([ih, 0]);
  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(1)).attr("y2", y(1)).attr("stroke", css("--axis")).attr("stroke-width", 1);
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 38).attr("text-anchor", "middle").text(xLabel);
  g.append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -40).attr("text-anchor", "middle").text(yLabel);
  const wins = points.filter((p) => p.y > p.x).length, losses = points.filter((p) => p.y < p.x).length;
  g.append("text").attr("class", "label-strong").attr("x", 8).attr("y", 14).text(`${wins} scans better with ours`);
  g.append("text").attr("class", "label").attr("x", iw - 8).attr("y", ih - 10).attr("text-anchor", "end").text(`${losses} better with the baseline`);
  g.append("g").selectAll("circle").data(points).join("circle")
    .attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 4)
    .attr("fill", (p) => (p.y >= p.x ? color : css("--muted-series"))).attr("fill-opacity", 0.85)
    .attr("stroke", css("--surface-1")).attr("stroke-width", 1.5);
  const delaunay = d3.Delaunay.from(points, (p) => x(p.x), (p) => y(p.y));
  g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [mx, my] = d3.pointer(event);
      const p = points[delaunay.find(mx, my)];
      if (p) showTip(event, p.id, [[xLabel, fmt3(p.x)], [yLabel, fmt3(p.y)], ["difference", d3.format("+.3f")(p.y - p.x)]]);
    })
    .on("mouseleave", hideTip);
}

// ---------------------------------------------------------------- strip of differences

export function strip(el, { values, label, height = 150, color }) {
  const margin = { top: 18, right: 18, bottom: 40, left: 18 };
  const { g, iw, ih } = frame(el, { height, margin });
  const ext = d3.extent(values.map((v) => v.value).concat([0]));
  const pad = (ext[1] - ext[0]) * 0.1 || 0.01;
  const x = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([0, iw]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("+.2f")).tickSizeOuter(0));
  g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", -6).attr("y2", ih).attr("stroke", css("--axis"));
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle").text(label);
  const sim = values.map((v, i) => ({ ...v, jitter: ((i * 37) % 17) / 17 - 0.5 }));
  g.append("g").selectAll("circle").data(sim).join("circle")
    .attr("cx", (v) => x(v.value)).attr("cy", (v) => ih / 2 + v.jitter * (ih * 0.7)).attr("r", 4)
    .attr("fill", (v) => (v.value >= 0 ? color : css("--muted-series"))).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5)
    .on("mousemove", (event, v) => showTip(event, v.id, [["difference", d3.format("+.3f")(v.value)]]))
    .on("mouseleave", hideTip);
  const mean = d3.mean(values, (v) => v.value);
  g.append("line").attr("x1", x(mean)).attr("x2", x(mean)).attr("y1", -6).attr("y2", ih).attr("stroke", color).attr("stroke-width", 2);
  g.append("text").attr("class", "label-strong").attr("x", x(mean) + 6).attr("y", -6 + 10).text(`mean ${d3.format("+.3f")(mean)}`);
}

// ---------------------------------------------------------------- dumbbell (per site)

export function dumbbell(el, { rows, aLabel, bLabel, aColor, bColor, xLabel, tip }) {
  const rowH = 22;
  const margin = { top: 6, right: 20, bottom: 40, left: 92 };
  const { g, iw, ih } = frame(el, { height: margin.top + margin.bottom + rows.length * rowH, margin });
  const x = d3.scaleLinear().domain([Math.max(0, d3.min(rows, (r) => Math.min(r.a, r.b)) - 0.05), Math.min(1, d3.max(rows, (r) => Math.max(r.a, r.b)) + 0.05)]).range([0, iw]);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, ih]).padding(0.3);
  g.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line").attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", 0).attr("y2", ih);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle").text(xLabel);
  const cy = (r) => y(r.label) + y.bandwidth() / 2;
  g.append("g").selectAll("line").data(rows).join("line").attr("x1", (r) => x(r.a)).attr("x2", (r) => x(r.b)).attr("y1", cy).attr("y2", cy).attr("stroke", css("--axis")).attr("stroke-width", 2);
  for (const [key, color] of [["b", bColor], ["a", aColor]]) {
    g.append("g").selectAll("circle").data(rows).join("circle").attr("cx", (r) => x(r[key])).attr("cy", cy).attr("r", 4.5)
      .attr("fill", color).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
  }
  g.append("g").selectAll("rect").data(rows).join("rect").attr("x", -margin.left).attr("width", iw + margin.left + margin.right)
    .attr("y", (r) => y(r.label) - 3).attr("height", y.bandwidth() + 6).attr("fill", "transparent")
    .on("mousemove", (event, r) => showTip(event, ...(tip ? tip(r) : [`${r.label} · ${r.n} subjects`, [[aLabel, fmt3(r.a)], [bLabel, fmt3(r.b)]]])))
    .on("mouseleave", hideTip);
  legend(el, [{ label: aLabel, color: aColor }, { label: bLabel, color: bColor }]);
}

// ---------------------------------------------------------------- reliability diagram

export function reliability(el, { series, height = 300 }) {
  const margin = { top: 12, right: 18, bottom: 44, left: 52 };
  const { g, iw, ih } = frame(el, { height, margin });
  const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((a) => a.select(".domain").remove());
  g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).attr("stroke", css("--axis"));
  g.append("text").attr("class", "label").attr("x", iw / 2).attr("y", ih + 38).attr("text-anchor", "middle").text("predicted probability of autism");
  g.append("text").attr("class", "label").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -40).attr("text-anchor", "middle").text("observed fraction");
  series.forEach((s) => {
    const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y));
    g.append("path").datum(s.points).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", line);
    g.append("g").selectAll("circle").data(s.points).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 4)
      .attr("fill", s.color).attr("stroke", css("--surface-1")).attr("stroke-width", 2)
      .on("mousemove", (event, p) => showTip(event, s.label, [["mean predicted", fmt3(p.x)], ["observed", fmt3(p.y)], ["subjects in bin", p.n]]))
      .on("mouseleave", hideTip);
  });
  legend(el, series.map((s) => ({ label: s.label, color: s.color })));
}

export { d3 };
