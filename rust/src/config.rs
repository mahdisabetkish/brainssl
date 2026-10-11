use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Contents of `segmentation.json`, written next to the ONNX file by
/// `brainssl.export.onnx_export`. Every constant the pipeline needs comes from
/// here, so the Python and Rust implementations cannot drift apart silently.
#[derive(Debug, Clone, Deserialize)]
pub struct SegmentationConfig {
    pub input: String,
    pub output: String,
    pub window: [usize; 3],
    pub overlap: f64,
    pub channels: Vec<String>,
    pub threshold: f32,
    pub lr_width: usize,
    pub crop_multiple: usize,
    pub crop_margin: usize,
    pub mirror_tta: bool,
    pub gaussian_sigma_scale: f64,
}

impl SegmentationConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("segmentation.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}
