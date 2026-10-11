// The connectome classifier in plain JavaScript, matching brainssl.fmri.final:
// Ledoit-Wolf correlation, matrix logarithm, sqrt(2)-scaled upper triangle,
// per-feature standardisation and a logistic score, averaged over the full scan
// and its windows. Matrices are Float64Array in row-major order.

export function shrunkCorrelation(ts, T, R) {
  // ts: Float32Array or Float64Array of length T*R, row-major (time, roi)
  const x = new Float64Array(T * R);
  for (let j = 0; j < R; j++) {
    let mean = 0;
    for (let t = 0; t < T; t++) mean += ts[t * R + j];
    mean /= T;
    let ss = 0;
    for (let t = 0; t < T; t++) { const v = ts[t * R + j] - mean; ss += v * v; }
    const std = Math.sqrt(ss / T);
    for (let t = 0; t < T; t++) x[t * R + j] = std < 1e-8 ? 0 : (ts[t * R + j] - mean) / std;
  }
  const s = new Float64Array(R * R);
  for (let t = 0; t < T; t++) {
    const row = t * R;
    for (let i = 0; i < R; i++) {
      const xi = x[row + i];
      if (xi === 0) continue;
      for (let j = i; j < R; j++) s[i * R + j] += xi * x[row + j];
    }
  }
  for (let i = 0; i < R; i++) for (let j = i; j < R; j++) { s[i * R + j] /= T; s[j * R + i] = s[i * R + j]; }
  let mu = 0;
  for (let i = 0; i < R; i++) mu += s[i * R + i];
  mu /= R;
  let delta = 0, sNorm = 0;
  for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
    const v = s[i * R + j];
    sNorm += v * v;
    const d = v - (i === j ? mu : 0);
    delta += d * d;
  }
  delta /= R;
  let beta = 0;
  for (let t = 0; t < T; t++) {
    let sq = 0;
    for (let j = 0; j < R; j++) sq += x[t * R + j] * x[t * R + j];
    beta += sq * sq;
  }
  beta = (beta / T - sNorm) / T / R;
  beta = Math.min(beta, delta);
  const shrink = delta > 0 ? beta / delta : 0;
  const cov = new Float64Array(R * R);
  for (let i = 0; i < R * R; i++) cov[i] = (1 - shrink) * s[i];
  for (let i = 0; i < R; i++) cov[i * R + i] += shrink * mu;
  const d = new Float64Array(R);
  for (let i = 0; i < R; i++) d[i] = Math.sqrt(Math.max(cov[i * R + i], 1e-12));
  for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) cov[i * R + j] /= d[i] * d[j];
  return cov;
}

// Cyclic Jacobi eigendecomposition of a symmetric matrix. Returns {values, vectors}
// with vectors stored column-wise (vectors[i*n + k] is component i of vector k).
export function jacobiEigen(a, n, tol = 1e-12, maxSweeps = 60) {
  const A = Float64Array.from(a);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}

export function spdLog(c, n) {
  const { values, vectors } = jacobiEigen(c, n);
  const out = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const l = Math.log(Math.max(values[k], 1e-12));
    for (let i = 0; i < n; i++) {
      const vik = vectors[i * n + k] * l;
      if (vik === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += vik * vectors[j * n + k];
    }
  }
  return out;
}

export function symToVec(m, n) {
  const out = new Float64Array((n * (n + 1)) / 2);
  let k = 0;
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) out[k++] = i === j ? m[i * n + j] : m[i * n + j] * Math.SQRT2;
  return out;
}

export function windowStarts(length, count, fraction, minLength = 40) {
  const width = Math.min(length, Math.max(minLength, Math.floor(fraction * length)));
  const span = length - width;
  const starts = [];
  for (let k = 0; k < count; k++) {
    const s = count > 1 ? (span * k) / (count - 1) : 0;
    starts.push(roundHalfEven(s));
  }
  return { starts, width };
}

function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function score(features, params, d, bias) {
  // params: Float32Array [mean(d) | std(d) | weight(d)]
  let logit = bias;
  for (let k = 0; k < d; k++) logit += params[2 * d + k] * ((features[k] - params[k]) / params[d + k]);
  return 1 / (1 + Math.exp(-logit));
}

export function classify(ts, T, R, model, onProgress) {
  const { params, bias, windows } = model;
  const d = (R * (R + 1)) / 2;
  const embed = (data, t) => symToVec(spdLog(shrunkCorrelation(data, t, R), R), R);
  const full = score(embed(ts, T), params, d, bias);
  onProgress?.(1, windows.count + 1);
  const { starts, width } = windowStarts(T, windows.count, windows.fraction, windows.min_length);
  const perWindow = starts.map((s, i) => {
    const p = score(embed(ts.subarray(s * R, (s + width) * R), width), params, d, bias);
    onProgress?.(i + 2, windows.count + 1);
    return p;
  });
  const meanWindows = perWindow.reduce((a, b) => a + b, 0) / perWindow.length;
  return { full, perWindow, starts, width, probability: 0.5 * (full + meanWindows) };
}
