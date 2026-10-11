//! Connectome classification from ROI time series.
//!
//! Mirrors `brainssl.fmri.final`: Ledoit-Wolf correlation, matrix logarithm,
//! the upper triangle with off-diagonal entries scaled by sqrt(2) (NumPy's
//! `triu_indices` order), per-feature standardisation and a logistic score,
//! averaged over the full scan and its windows.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Layout {
    mean: [usize; 2],
    std: [usize; 2],
    weight: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct AtlasModel {
    n_rois: usize,
    n_features: usize,
    bias: f64,
    file: String,
    layout: Layout,
}

#[derive(Debug, Deserialize)]
struct Windows {
    count: usize,
    fraction: f64,
    min_length: usize,
}

#[derive(Debug, Deserialize)]
struct ConnectomeModel {
    windows: Windows,
    atlases: BTreeMap<String, AtlasModel>,
}

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub atlas: String,
    pub n_timepoints: usize,
    pub n_rois: usize,
    pub probability_full_scan: f64,
    pub probability_windows: f64,
    pub probability: f64,
    pub label: &'static str,
}

pub fn read_timeseries(path: &Path) -> Result<DMatrix<f64>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| l.split_whitespace().map(|v| v.parse::<f64>()).collect::<Result<Vec<_>, _>>())
        .collect::<Result<_, _>>()
        .context("parsing the time series")?;
    let t = rows.len();
    let r = rows.first().map(|v| v.len()).unwrap_or(0);
    if t < 2 || r == 0 || rows.iter().any(|v| v.len() != r) {
        bail!("expected a rectangular table with at least two time points");
    }
    Ok(DMatrix::from_fn(t, r, |i, j| rows[i][j]))
}

/// Ledoit-Wolf shrunk correlation of z-scored series, as in `connectivity.py`.
pub fn shrunk_correlation(ts: &DMatrix<f64>) -> DMatrix<f64> {
    let (t, r) = ts.shape();
    let mut x = ts.clone();
    for j in 0..r {
        let mut col = x.column_mut(j);
        let mean = col.mean();
        col.add_scalar_mut(-mean);
        let std = (col.iter().map(|v| v * v).sum::<f64>() / t as f64).sqrt();
        if std < 1e-8 {
            col.fill(0.0);
        } else {
            col /= std;
        }
    }
    let n = t as f64;
    let s = x.transpose() * &x / n;
    let mu = s.trace() / r as f64;
    let mut delta = 0.0;
    for i in 0..r {
        for j in 0..r {
            let v = s[(i, j)] - if i == j { mu } else { 0.0 };
            delta += v * v;
        }
    }
    delta /= r as f64;
    let mut beta = 0.0;
    for row in x.row_iter() {
        let sq: f64 = row.iter().map(|v| v * v).sum();
        beta += sq * sq;
    }
    let s_norm: f64 = s.iter().map(|v| v * v).sum();
    beta = (beta / n - s_norm) / n / r as f64;
    let beta = beta.min(delta);
    let shrink = if delta > 0.0 { beta / delta } else { 0.0 };
    let mut cov = s * (1.0 - shrink);
    for i in 0..r {
        cov[(i, i)] += shrink * mu;
    }
    let d: Vec<f64> = (0..r).map(|i| cov[(i, i)].max(1e-12).sqrt()).collect();
    DMatrix::from_fn(r, r, |i, j| cov[(i, j)] / (d[i] * d[j]))
}

/// Matrix logarithm of a symmetric positive definite matrix.
pub fn spd_log(c: &DMatrix<f64>) -> DMatrix<f64> {
    let sym = (c + c.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let logs = eig.eigenvalues.map(|v| v.max(1e-12).ln());
    &eig.eigenvectors * DMatrix::from_diagonal(&logs) * eig.eigenvectors.transpose()
}

/// Upper triangle including the diagonal, row by row, off-diagonal times sqrt(2).
pub fn sym_to_vec(m: &DMatrix<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n {
            out.push(if i == j { m[(i, j)] } else { m[(i, j)] * std::f64::consts::SQRT_2 });
        }
    }
    out
}

fn window_starts(length: usize, count: usize, fraction: f64, min_length: usize) -> (Vec<usize>, usize) {
    let width = length.min(min_length.max((fraction * length as f64) as usize));
    let span = (length - width) as f64;
    let starts = (0..count)
        .map(|k| {
            let s = if count > 1 { span * k as f64 / (count - 1) as f64 } else { 0.0 };
            s.round_ties_even() as usize
        })
        .collect();
    (starts, width)
}

fn read_f32(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} is not a float32 array", path.display());
    }
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b) as f64).collect())
}

fn score(features: &[f64], params: &[f64], model: &AtlasModel) -> f64 {
    let (m, s, w) = (&model.layout.mean, &model.layout.std, &model.layout.weight);
    let mut logit = model.bias;
    for (k, &x) in features.iter().enumerate() {
        let z = (x - params[m[0] + k]) / params[s[0] + k];
        logit += params[w[0] + k] * z;
    }
    1.0 / (1.0 + (-logit).exp())
}

pub fn classify(model_dir: &Path, timeseries: &Path, atlas: &str, _threads: usize) -> Result<Prediction> {
    let text = std::fs::read_to_string(model_dir.join("connectome.json")).context("reading connectome.json")?;
    let cfg: ConnectomeModel = serde_json::from_str(&text).context("parsing connectome.json")?;
    let model = cfg
        .atlases
        .get(atlas)
        .with_context(|| format!("atlas {atlas} is not in this model; available: {:?}", cfg.atlases.keys()))?;
    let params = read_f32(&model_dir.join(&model.file))?;
    if params.len() != 3 * model.n_features {
        bail!("{} holds {} values, expected {}", model.file, params.len(), 3 * model.n_features);
    }
    let ts = read_timeseries(timeseries)?;
    if ts.ncols() != model.n_rois {
        bail!("{atlas} has {} ROIs but the file has {} columns", model.n_rois, ts.ncols());
    }
    let embed = |x: &DMatrix<f64>| sym_to_vec(&spd_log(&shrunk_correlation(x)));
    let p_full = score(&embed(&ts), &params, model);
    let (starts, width) = window_starts(ts.nrows(), cfg.windows.count, cfg.windows.fraction, cfg.windows.min_length);
    let p_windows = starts
        .iter()
        .map(|&s| score(&embed(&ts.rows(s, width).into_owned()), &params, model))
        .sum::<f64>()
        / starts.len() as f64;
    let p = 0.5 * (p_full + p_windows);
    Ok(Prediction {
        atlas: atlas.to_string(),
        n_timepoints: ts.nrows(),
        n_rois: ts.ncols(),
        probability_full_scan: p_full,
        probability_windows: p_windows,
        probability: p,
        label: if p >= 0.5 { "autism (ASD)" } else { "typically developing control" },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_has_unit_diagonal_and_is_symmetric() {
        let ts = DMatrix::from_fn(50, 6, |i, j| ((i * (j + 1)) as f64 * 0.37).sin() + j as f64 * 0.01 * i as f64);
        let c = shrunk_correlation(&ts);
        for i in 0..6 {
            assert!((c[(i, i)] - 1.0).abs() < 1e-12);
            for j in 0..6 {
                assert!((c[(i, j)] - c[(j, i)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn log_of_diagonal_matrix() {
        let c = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 0.5]);
        let l = spd_log(&c);
        assert!((l[(0, 0)] - 2f64.ln()).abs() < 1e-12);
        assert!((l[(1, 1)] - 0.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn vectorisation_order_and_isometry() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]);
        let v = sym_to_vec(&m);
        let s = std::f64::consts::SQRT_2;
        assert_eq!(v, vec![1.0, 2.0 * s, 3.0 * s, 4.0, 5.0 * s, 6.0]);
        let frob: f64 = m.iter().map(|x| x * x).sum::<f64>();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>();
        assert!((frob - norm).abs() < 1e-9);
    }

    #[test]
    fn windows_match_python() {
        // window_starts(196, 6, 0.7) in Python gives width 137 and these starts.
        let (starts, width) = window_starts(196, 6, 0.7, 40);
        assert_eq!(width, 137);
        assert_eq!(starts, vec![0, 12, 24, 35, 47, 59]);
    }
}
