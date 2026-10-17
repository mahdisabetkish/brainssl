//! brainssl-infer: run the exported brainssl models without Python.
//!
//! The binary links ONNX Runtime at run time (`--ort-lib` or `ORT_DYLIB_PATH`),
//! so the same build works with a CPU or a GPU build of the library.

mod config;
mod connectome;
mod preprocess;
mod sliding;
mod volume;

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ort::session::Session;

#[derive(Parser)]
#[command(name = "brainssl-infer", version, about = "Inference for the brainssl models without Python")]
struct Cli {
    /// ONNX Runtime shared library (onnxruntime.dll, libonnxruntime.so or .dylib).
    /// Falls back to the ORT_DYLIB_PATH environment variable.
    #[arg(long, global = true)]
    ort_lib: Option<PathBuf>,
    /// Intra-op threads for ONNX Runtime; 0 lets the runtime decide.
    #[arg(long, global = true, default_value_t = 0)]
    threads: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Segment a brain tumour in a four-contrast MRI volume (FLAIR, T1, T1Gd, T2).
    Segment {
        /// Folder with segmentation.onnx and segmentation.json.
        #[arg(long)]
        model: PathBuf,
        /// Input NIfTI, 4D with the contrasts on the last axis (MSD layout).
        #[arg(long)]
        input: PathBuf,
        /// Output label map (0 background, 1 oedema, 2 non-enhancing core, 3 enhancing).
        #[arg(long)]
        output: PathBuf,
        /// Also write the three region probabilities as a 4D NIfTI.
        #[arg(long)]
        probabilities: Option<PathBuf>,
        /// Skip the left-right mirrored second pass.
        #[arg(long)]
        no_mirror: bool,
    },
    /// Classify a resting-state scan from its ROI time series.
    Classify {
        /// Folder with connectome.onnx and connectome.json.
        #[arg(long)]
        model: PathBuf,
        /// Whitespace-separated time series, one column per ROI (the PCP .1D format).
        #[arg(long)]
        timeseries: PathBuf,
        /// Atlas the time series were extracted with, e.g. cc200.
        #[arg(long)]
        atlas: String,
    },
}

fn init_runtime(cli: &Cli) -> Result<()> {
    let lib = cli
        .ort_lib
        .clone()
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
        .context("no ONNX Runtime library given; pass --ort-lib or set ORT_DYLIB_PATH")?;
    ort::init_from(&lib)
        .with_context(|| format!("loading {}", lib.display()))?
        .commit();
    Ok(())
}

fn session(path: &std::path::Path, threads: usize) -> Result<Session> {
    // Builder errors hand the builder back for recovery and are not Send, so
    // they are flattened to messages here.
    let err = |e: &dyn std::fmt::Display| anyhow::anyhow!("{e}");
    let mut builder = Session::builder().map_err(|e| err(&e))?;
    if threads > 0 {
        builder = builder.with_intra_threads(threads).map_err(|e| err(&e))?;
    }
    builder
        .commit_from_file(path)
        .map_err(|e| err(&e))
        .with_context(|| format!("loading {}", path.display()))
}

fn main() -> Result<()> {
    run()?;
    // The ONNX Runtime library is loaded at run time. Leaving through exit skips its teardown, which
    // ends in a crash on some Linux builds after all the work and output are done.
    std::io::stdout().flush()?;
    std::process::exit(0)
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    init_runtime(&cli)?;
    match &cli.command {
        Command::Segment { model, input, output, probabilities, no_mirror } => {
            let start = Instant::now();
            let cfg = config::SegmentationConfig::load(model)?;
            let mut session = session(&model.join("segmentation.onnx"), cli.threads)?;
            let vol = volume::read_multicontrast(input)?;
            let (channels, nx, ny, nz) = vol.data.dim();
            if channels != cfg.channels.len() {
                anyhow::bail!(
                    "{} has {channels} contrasts on its last axis; the model expects {} ({})",
                    input.display(),
                    cfg.channels.len(),
                    cfg.channels.join(", ")
                );
            }
            let mask = preprocess::brain_mask(vol.data.view());
            let midline = preprocess::estimate_midline(&mask, 30);
            let bbox = preprocess::crop_box(&mask, midline, cfg.lr_width, cfg.crop_multiple, cfg.crop_margin);
            let mut image = preprocess::take(vol.data.view(), &bbox);
            preprocess::normalize(&mut image);
            eprintln!("midline at x = {midline:.2}, crop {:?}", image.dim());
            let mirror = cfg.mirror_tta && !no_mirror;
            let prob = sliding::predict(&mut session, &cfg, image.view(), mirror, |i, n| {
                if i < n {
                    eprint!("\rwindow {}/{}", i + 1, n);
                } else {
                    eprintln!();
                }
            })?;
            let labels = sliding::labels_from_probabilities(&prob, cfg.threshold);
            let full = preprocess::uncrop_labels(&labels, &bbox, (nx, ny, nz));
            volume::write_labels(output, &full, &vol.header)?;
            if let Some(path) = probabilities {
                let full = preprocess::uncrop_prob(&prob, &bbox, (nx, ny, nz));
                volume::write_probabilities(path, &full, &vol.header)?;
            }
            let counts: Vec<usize> = (1..=3u8).map(|l| full.iter().filter(|&&v| v == l).count()).collect();
            eprintln!(
                "voxels: oedema {}, non-enhancing {}, enhancing {} ({:.1}s)",
                counts[0], counts[1], counts[2], start.elapsed().as_secs_f64()
            );
            // dropping the session unloads the ONNX Runtime library, which crashes on some Linux builds
            std::mem::forget(session);
        }
        Command::Classify { model, timeseries, atlas } => {
            let result = connectome::classify(model, timeseries, atlas, cli.threads)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}
