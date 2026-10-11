# Inference outside Python

## The Rust binary

`rust/` builds `brainssl-infer`, a single executable that runs both models. It
links ONNX Runtime at run time (pass `--ort-lib` or set `ORT_DYLIB_PATH`), so the
same build works with a CPU or a GPU build of the runtime.

```bash
cargo build --release --manifest-path rust/Cargo.toml

# tumour segmentation of a four-contrast NIfTI (contrasts on the last axis)
brainssl-infer --ort-lib /opt/onnxruntime/lib/libonnxruntime.so segment \
    --model exports/segmentation --input scan.nii.gz --output scan_seg.nii.gz \
    --probabilities scan_prob.nii.gz

# connectome classification from ROI time series (PCP .1D format)
brainssl-infer --ort-lib /opt/onnxruntime/lib/libonnxruntime.so classify \
    --model exports/connectome --timeseries subject_rois_cc200.1D --atlas cc200
```

Everything the Python pipeline does around the network is reimplemented, not
approximated: the midline search, the symmetric crop, per-contrast
normalisation, window placement, Gaussian blending, mirror averaging and the
region hierarchy for segmentation; the Ledoit-Wolf estimator, the matrix
logarithm, the windows and the intercept for classification. All constants are
read from the JSON file exported next to each model, so the two implementations
cannot drift apart without a test noticing.

## Parity

`scripts/verify_rust_parity.py` runs both implementations on the same scan with
the same exported model and compares the midline estimate, the probabilities and
the label maps. On a full 240 x 240 x 155 MSD scan the largest probability
difference was 3.4e-6 and 99.99996% of voxels received the same label (the four
that did not sit exactly on the 0.5 threshold); the Rust binary took 33 s on the
CPU against 79 s for Python with ONNX Runtime. CI repeats the check on every
change, with a small randomly initialised network on synthetic data.

## ONNX

`brainssl export segmentation` writes `segmentation.onnx` for one sliding
window, input `image` of shape (N, 4, 160, 128, 128) and output `logits` of the
same spatial size with three channels, and checks the exported graph against
PyTorch before writing. The bilateral blocks export as slices with a negative
step, so no custom operator is needed.

## In the browser

The dashboard runs the connectome classifier in plain JavaScript inside a Web
Worker, on held-out subjects of one cross-validation split, and runs the
segmentation network with ONNX Runtime Web (WebGPU when the browser offers it,
WebAssembly otherwise), fetching the model from the Hugging Face Hub.

## Hugging Face Hub

`brainssl export all` writes two folders with model cards whose numbers are read
from the report. `scripts/push_to_hub.py` uploads them as
`<user>/brainssl-tumour-segmentation` and `<user>/brainssl-abide-connectome`,
and the manual `Hugging Face` workflow does the same from a GitHub release.
