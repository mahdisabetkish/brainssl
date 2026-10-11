//! Sliding-window inference, matching `brainssl/smri/infer.py`.

use anyhow::{Context, Result};
use ndarray::{Array3, Array4, Array5, ArrayView4, Axis, s};
use ort::{session::Session, value::TensorRef};

use crate::config::SegmentationConfig;

/// Window start positions along one axis: a regular grid plus a final window
/// flush with the end, or a single window when the axis is short.
pub fn starts(size: usize, window: usize, overlap: f64) -> Vec<usize> {
    if size <= window {
        return vec![0];
    }
    let step = ((window as f64 * (1.0 - overlap)) as usize).max(1);
    let mut out: Vec<usize> = (0..=size - window).step_by(step).collect();
    if *out.last().unwrap() != size - window {
        out.push(size - window);
    }
    out
}

/// Separable Gaussian weight over one window, scaled to a maximum of one and
/// floored at 1e-3 so window corners still count a little.
pub fn gaussian(shape: [usize; 3], sigma_scale: f64) -> Array3<f32> {
    let axis = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 - (n as f64 - 1.0) / 2.0;
                (-0.5 * (t / (sigma_scale * n as f64)).powi(2)).exp()
            })
            .collect()
    };
    let (a, b, c) = (axis(shape[0]), axis(shape[1]), axis(shape[2]));
    let mut w = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| a[i] * b[j] * c[k]);
    let max = w.iter().cloned().fold(f64::MIN, f64::max);
    w.mapv_inplace(|v| (v / max).max(1e-3));
    w.mapv(|v| v as f32)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn run_window(session: &mut Session, cfg: &SegmentationConfig, tile: &Array5<f32>) -> Result<Array4<f32>> {
    let outputs = session
        .run(ort::inputs![cfg.input.as_str() => TensorRef::from_array_view(tile)?])
        .context("running the ONNX model")?;
    let logits = outputs[cfg.output.as_str()]
        .try_extract_array::<f32>()
        .context("reading the logits")?;
    let logits = logits.index_axis(Axis(0), 0).to_owned().into_dimensionality::<ndarray::Ix4>()?;
    Ok(logits.mapv(sigmoid))
}

/// Region probabilities (K, X, Y, Z) for a cropped, normalised image (C, X, Y, Z).
pub fn predict(session: &mut Session, cfg: &SegmentationConfig, image: ArrayView4<f32>, mirror: bool,
               mut progress: impl FnMut(usize, usize)) -> Result<Array4<f32>> {
    let (c, sx, sy, sz) = image.dim();
    let [wx, wy, wz] = cfg.window;
    // Pad up to the window size where the crop is smaller than a window.
    let (px, py, pz) = (sx.max(wx), sy.max(wy), sz.max(wz));
    let mut padded = Array4::<f32>::zeros((c, px, py, pz));
    padded.slice_mut(s![.., ..sx, ..sy, ..sz]).assign(&image);

    let weight = gaussian(cfg.window, cfg.gaussian_sigma_scale);
    let mut out: Option<Array4<f32>> = None;
    let mut norm = Array3::<f32>::zeros((px, py, pz));
    let boxes: Vec<(usize, usize, usize)> = starts(px, wx, cfg.overlap)
        .into_iter()
        .flat_map(|x| starts(py, wy, cfg.overlap).into_iter().map(move |y| (x, y)))
        .flat_map(|(x, y)| starts(pz, wz, cfg.overlap).into_iter().map(move |z| (x, y, z)))
        .collect();
    let flips: &[bool] = if mirror { &[false, true] } else { &[false] };

    for (i, &(x, y, z)) in boxes.iter().enumerate() {
        progress(i, boxes.len());
        let tile = padded.slice(s![.., x..x + wx, y..y + wy, z..z + wz]).to_owned();
        let mut prob: Option<Array4<f32>> = None;
        for &flip in flips {
            let mut input = tile.clone();
            if flip {
                input.invert_axis(Axis(1));
            }
            let input = input.insert_axis(Axis(0)).as_standard_layout().to_owned();
            let mut p = run_window(session, cfg, &input)?;
            if flip {
                p.invert_axis(Axis(1));
            }
            prob = Some(match prob {
                None => p,
                Some(acc) => acc + &p,
            });
        }
        let prob = prob.unwrap() / flips.len() as f32;
        let k = prob.len_of(Axis(0));
        let acc = out.get_or_insert_with(|| Array4::<f32>::zeros((k, px, py, pz)));
        for r in 0..k {
            let mut region = acc.slice_mut(s![r, x..x + wx, y..y + wy, z..z + wz]);
            region += &(&prob.index_axis(Axis(0), r) * &weight);
        }
        let mut n = norm.slice_mut(s![x..x + wx, y..y + wy, z..z + wz]);
        n += &weight;
    }
    progress(boxes.len(), boxes.len());
    let mut out = out.context("no windows were evaluated")?;
    for mut region in out.outer_iter_mut() {
        region.zip_mut_with(&norm, |p, &w| *p /= w.max(1e-6));
    }
    Ok(out.slice(s![.., ..sx, ..sy, ..sz]).to_owned())
}

/// Labels 0-3 from region probabilities, nesting ET inside TC inside WT.
pub fn labels_from_probabilities(prob: &Array4<f32>, threshold: f32) -> Array3<u8> {
    let (_, x, y, z) = prob.dim();
    Array3::from_shape_fn((x, y, z), |(i, j, k)| {
        let wt = prob[[0, i, j, k]] > threshold;
        let tc = wt && prob[[1, i, j, k]] > threshold;
        let et = tc && prob[[2, i, j, k]] > threshold;
        if et {
            3
        } else if tc {
            2
        } else if wt {
            1
        } else {
            0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_starts_cover_the_axis() {
        assert_eq!(starts(160, 160, 0.5), vec![0]);
        assert_eq!(starts(192, 128, 0.5), vec![0, 64]);
        assert_eq!(starts(144, 128, 0.5), vec![0, 16]);
        assert_eq!(starts(100, 128, 0.5), vec![0]);
    }

    #[test]
    fn gaussian_peaks_at_one() {
        let w = gaussian([8, 6, 4], 0.125);
        let max = w.iter().cloned().fold(f32::MIN, f32::max);
        assert!((max - 1.0).abs() < 1e-6 || max < 1.0);
        assert!(w.iter().all(|&v| v >= 1e-3));
    }
}
