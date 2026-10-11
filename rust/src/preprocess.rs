//! Cropping and normalisation, matching `brainssl/smri/preprocess.py`.

use ndarray::{Array3, Array4, ArrayView4, Axis, s};

/// Half-open index range along one axis; `start` may be negative and `stop`
/// may run past the volume, in which case the crop is zero-padded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: i64,
    pub stop: i64,
}

impl Span {
    pub fn len(&self) -> usize {
        (self.stop - self.start) as usize
    }
}

pub fn brain_mask(image: ArrayView4<f32>) -> Array3<bool> {
    let (_, nx, ny, nz) = image.dim();
    let mut mask = Array3::from_elem((nx, ny, nz), false);
    for c in 0..image.len_of(Axis(0)) {
        ndarray::Zip::from(&mut mask)
            .and(image.index_axis(Axis(0), c))
            .for_each(|m, &v| *m |= v > 0.0);
    }
    mask
}

/// Reflection centre along X such that x -> 2c - x maps the mask onto itself.
///
/// Same estimator as the Python version: overlap of the mask with its mirror
/// for every integer shift within +-30 voxels, then a parabola through the
/// best shift and its two neighbours.
pub fn estimate_midline(mask: &Array3<bool>, max_shift: i64) -> f64 {
    let (nx, ny, nz) = mask.dim();
    let n = nx as i64;
    // Column masks along X for each (y, z) line, so the inner loop is a dot product.
    let lines: Vec<Vec<u8>> = (0..ny)
        .flat_map(|y| (0..nz).map(move |z| (y, z)))
        .map(|(y, z)| (0..nx).map(|x| mask[[x, y, z]] as u8).collect::<Vec<u8>>())
        .filter(|line| line.iter().any(|&v| v != 0))
        .collect();
    let shifts: Vec<i64> = (-max_shift..=max_shift).collect();
    let overlap: Vec<f64> = shifts
        .iter()
        .map(|&s| {
            let mut total = 0u64;
            for line in &lines {
                for u in 0..n {
                    let v = n - 1 + s - u;
                    if v >= 0 && v < n {
                        total += (line[u as usize] & line[v as usize]) as u64;
                    }
                }
            }
            total as f64
        })
        .collect();
    let (i, _) = overlap
        .iter()
        .enumerate()
        .fold((0usize, f64::MIN), |best, (k, &v)| if v > best.1 { (k, v) } else { best });
    let mut offset = 0.0;
    if i > 0 && i + 1 < overlap.len() {
        let (a, b, c) = (overlap[i - 1], overlap[i], overlap[i + 1]);
        let denom = a - 2.0 * b + c;
        if denom != 0.0 {
            offset = 0.5 * (a - c) / denom;
        }
    }
    (n as f64 - 1.0 + shifts[i] as f64 + offset) / 2.0
}

/// Bounding box of the mask along Y and Z, as half-open ranges.
fn bounds(mask: &Array3<bool>, axis: usize) -> (i64, i64) {
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for ((x, y, z), &m) in mask.indexed_iter() {
        if m {
            let i = [x, y, z][axis] as i64;
            lo = lo.min(i);
            hi = hi.max(i + 1);
        }
    }
    (lo, hi)
}

/// Python's `round`, which rounds halves to the nearest even integer.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub fn crop_box(mask: &Array3<bool>, midline: f64, lr_width: usize, multiple: usize, margin: usize) -> [Span; 3] {
    let x0 = py_round(midline - (lr_width as f64 - 1.0) / 2.0);
    let mut out = [Span { start: x0, stop: x0 + lr_width as i64 }; 3];
    for (k, axis) in [1usize, 2].into_iter().enumerate() {
        let (lo, hi) = bounds(mask, axis);
        let (lo, hi) = (lo - margin as i64, hi + margin as i64);
        let m = multiple as i64;
        let size = (hi - lo + m - 1) / m * m;
        let centre = (lo + hi).div_euclid(2);
        let start = centre - size / 2;
        out[k + 1] = Span { start, stop: start + size };
    }
    out
}

/// Crop (C, X, Y, Z) with zero padding where the box leaves the volume.
pub fn take(image: ArrayView4<f32>, b: &[Span; 3]) -> Array4<f32> {
    let (c, nx, ny, nz) = image.dim();
    let mut out = Array4::<f32>::zeros((c, b[0].len(), b[1].len(), b[2].len()));
    let dims = [nx as i64, ny as i64, nz as i64];
    let src: Vec<(i64, i64)> = (0..3).map(|k| (b[k].start.max(0), b[k].stop.min(dims[k]))).collect();
    if src.iter().any(|(a, z)| a >= z) {
        return out;
    }
    let dst: Vec<(i64, i64)> = (0..3).map(|k| (src[k].0 - b[k].start, src[k].1 - b[k].start)).collect();
    let (d, r) = (ranges(&dst), ranges(&src));
    out.slice_mut(s![.., d[0].clone(), d[1].clone(), d[2].clone()])
        .assign(&image.slice(s![.., r[0].clone(), r[1].clone(), r[2].clone()]));
    out
}

fn ranges(bounds: &[(i64, i64)]) -> Vec<std::ops::Range<usize>> {
    bounds.iter().map(|&(a, b)| a as usize..b as usize).collect()
}

/// Inverse of `take` for a label map: place the crop back on the full grid.
pub fn uncrop_labels(labels: &Array3<u8>, b: &[Span; 3], shape: (usize, usize, usize)) -> Array3<u8> {
    let mut out = Array3::<u8>::zeros(shape);
    let dims = [shape.0 as i64, shape.1 as i64, shape.2 as i64];
    let dst: Vec<(i64, i64)> = (0..3).map(|k| (b[k].start.max(0), b[k].stop.min(dims[k]))).collect();
    let src: Vec<(i64, i64)> = (0..3).map(|k| (dst[k].0 - b[k].start, dst[k].1 - b[k].start)).collect();
    let (d, r) = (ranges(&dst), ranges(&src));
    out.slice_mut(s![d[0].clone(), d[1].clone(), d[2].clone()])
        .assign(&labels.slice(s![r[0].clone(), r[1].clone(), r[2].clone()]));
    out
}

/// Same as `uncrop_labels` for (K, X, Y, Z) probabilities.
pub fn uncrop_prob(prob: &Array4<f32>, b: &[Span; 3], shape: (usize, usize, usize)) -> Array4<f32> {
    let k = prob.len_of(Axis(0));
    let mut out = Array4::<f32>::zeros((k, shape.0, shape.1, shape.2));
    let dims = [shape.0 as i64, shape.1 as i64, shape.2 as i64];
    let dst: Vec<(i64, i64)> = (0..3).map(|i| (b[i].start.max(0), b[i].stop.min(dims[i]))).collect();
    let src: Vec<(i64, i64)> = (0..3).map(|i| (dst[i].0 - b[i].start, dst[i].1 - b[i].start)).collect();
    let (d, r) = (ranges(&dst), ranges(&src));
    out.slice_mut(s![.., d[0].clone(), d[1].clone(), d[2].clone()])
        .assign(&prob.slice(s![.., r[0].clone(), r[1].clone(), r[2].clone()]));
    out
}

/// Z-score each contrast over voxels where any contrast is non-zero.
pub fn normalize(image: &mut Array4<f32>) {
    let mask = brain_mask(image.view());
    let count = mask.iter().filter(|&&m| m).count().max(1) as f64;
    for c in 0..image.len_of(Axis(0)) {
        let mut channel = image.index_axis_mut(Axis(0), c);
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        ndarray::Zip::from(&channel).and(&mask).for_each(|&v, &m| {
            if m {
                sum += v as f64;
                sq += (v as f64) * (v as f64);
            }
        });
        let mean = sum / count;
        let std = (sq / count - mean * mean).max(0.0).sqrt();
        ndarray::Zip::from(&mut channel).and(&mask).for_each(|v, &m| {
            *v = if m { ((*v as f64 - mean) / (std + 1e-6)) as f32 } else { 0.0 };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ellipsoid(cx: f64) -> Array3<bool> {
        Array3::from_shape_fn((240, 240, 155), |(x, y, z)| {
            let (x, y, z) = (x as f64, y as f64, z as f64);
            ((x - cx) / 65.0).powi(2) + ((y - 120.0) / 85.0).powi(2) + ((z - 75.0) / 60.0).powi(2) <= 1.0
        })
    }

    #[test]
    fn midline_of_symmetric_shapes() {
        for cx in [117.0, 119.5, 122.0] {
            let c = estimate_midline(&ellipsoid(cx), 30);
            assert!((c - cx).abs() < 0.3, "expected {cx}, got {c}");
        }
    }

    #[test]
    fn crop_is_centred_on_the_midline() {
        let mask = ellipsoid(119.5);
        let b = crop_box(&mask, 119.5, 160, 16, 2);
        assert_eq!(b[0].len(), 160);
        assert_eq!(b[0].start + b[0].stop - 1, 239);
        assert!(b.iter().all(|s| s.len() % 16 == 0));
    }

    #[test]
    fn python_rounding_of_halves() {
        assert_eq!(py_round(40.5), 40);
        assert_eq!(py_round(41.5), 42);
        assert_eq!(py_round(-0.5), 0);
    }
}
