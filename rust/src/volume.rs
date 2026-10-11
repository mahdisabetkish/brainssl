use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array3, Array4, Axis, Ix4};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions, writer::WriterOptions};

/// A multi-contrast MRI volume, channels first: (C, X, Y, Z).
pub struct Volume {
    pub data: Array4<f32>,
    pub header: NiftiHeader,
}

/// Read a 4D NIfTI whose last axis holds the contrasts (the MSD layout).
pub fn read_multicontrast(path: &Path) -> Result<Volume> {
    let object = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let header = object.header().clone();
    let array = object
        .into_volume()
        .into_ndarray::<f32>()
        .context("converting the volume to f32")?;
    let array = array
        .into_dimensionality::<Ix4>()
        .context("expected a 4D volume (X, Y, Z, contrast)")?;
    if array.shape()[3] != 4 {
        bail!("expected 4 contrasts on the last axis, found {}", array.shape()[3]);
    }
    // (X, Y, Z, C) -> (C, X, Y, Z), copied into standard row-major layout.
    let data = array.permuted_axes([3, 0, 1, 2]).as_standard_layout().to_owned();
    Ok(Volume { data, header })
}

/// Write a label map using the input's header for orientation and spacing.
pub fn write_labels(path: &Path, labels: &Array3<u8>, reference: &NiftiHeader) -> Result<()> {
    let mut header = reference.clone();
    header.dim[0] = 3;
    header.dim[4] = 1;
    header.pixdim[4] = 1.0;
    WriterOptions::new(path)
        .reference_header(&header)
        .write_nifti(labels)
        .with_context(|| format!("writing {}", path.display()))
}

/// Write per-region probabilities as a 4D float volume (X, Y, Z, region).
pub fn write_probabilities(path: &Path, prob: &Array4<f32>, reference: &NiftiHeader) -> Result<()> {
    let mut header = reference.clone();
    header.dim[4] = prob.len_of(Axis(0)) as u16;
    let xyzc = prob.view().permuted_axes([1, 2, 3, 0]).as_standard_layout().to_owned();
    WriterOptions::new(path)
        .reference_header(&header)
        .write_nifti(&xyzc)
        .with_context(|| format!("writing {}", path.display()))
}
