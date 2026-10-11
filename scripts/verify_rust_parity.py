"""Check that the Rust binary reproduces the Python segmentation pipeline.

Runs both on the same NIfTI volume with the same exported model and compares
the midline estimate, the crop and the region probabilities. With trained or
random weights alike, the two should agree to float32 round-off.

    python scripts/verify_rust_parity.py --model exports/segmentation \
        --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from brainssl.smri.infer import enforce_hierarchy, sliding_window
from brainssl.smri.preprocess import (
    _take,
    crop_box,
    estimate_midline,
    labels_from_regions,
    normalize,
    read_volume,
    uncrop,
)


class OnnxModel(torch.nn.Module):
    """Wrap an ONNX session so ``sliding_window`` can drive it like a torch model."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.anchor = torch.nn.Parameter(torch.zeros(1))      # gives the wrapper a device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.session.run(["logits"], {"image": x.cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(out)


def python_reference(model_dir: Path, image_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    raw = read_volume(image_path)
    mask = (raw > 0).any(axis=0)
    idx = np.argwhere(mask)
    row = pd.Series({"midline": estimate_midline(mask), "y0": idx[:, 1].min(), "y1": idx[:, 1].max() + 1,
                     "z0": idx[:, 2].min(), "z1": idx[:, 2].max() + 1})
    cfg = json.loads((model_dir / "segmentation.json").read_text())
    box = crop_box(row, shape=raw.shape[1:], lr_width=cfg["lr_width"], multiple=cfg["crop_multiple"],
                   margin=cfg["crop_margin"])
    image = normalize(_take(raw, box))
    prob = sliding_window(OnnxModel(model_dir / "segmentation.onnx"), torch.from_numpy(image),
                          window=tuple(cfg["window"]), overlap=cfg["overlap"], mirror=cfg["mirror_tta"])
    prob = prob.numpy()
    labels = labels_from_regions(enforce_hierarchy(prob, cfg["threshold"]))
    return uncrop(prob, box, raw.shape[1:]), uncrop(labels, box, raw.shape[1:]), float(row["midline"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("rust/target/release/brainssl-infer"))
    parser.add_argument("--ort-lib", type=Path, default=None,
                        help="ONNX Runtime shared library; defaults to the one bundled with onnxruntime")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    ort_lib = args.ort_lib
    if ort_lib is None:
        capi = Path(ort.__file__).parent / "capi"
        candidates = sorted(capi.glob("onnxruntime.dll")) + sorted(capi.glob("libonnxruntime.so*")) \
            + sorted(capi.glob("libonnxruntime*.dylib"))
        if not candidates:
            sys.exit("could not find the ONNX Runtime library; pass --ort-lib")
        ort_lib = candidates[0]

    t = time.time()
    prob_py, labels_py, midline_py = python_reference(args.model, args.input)
    t_py = time.time() - t

    with tempfile.TemporaryDirectory() as tmp:
        out_labels, out_prob = Path(tmp) / "labels.nii.gz", Path(tmp) / "prob.nii.gz"
        binary = args.binary.with_suffix(".exe") if os.name == "nt" else args.binary
        t = time.time()
        run = subprocess.run([str(binary), "--ort-lib", str(ort_lib), "segment", "--model", str(args.model),
                              "--input", str(args.input), "--output", str(out_labels),
                              "--probabilities", str(out_prob)], capture_output=True, text=True)
        t_rs = time.time() - t
        if run.returncode != 0:
            sys.exit(f"rust binary failed:\n{run.stderr}")
        midline_rs = float(run.stderr.split("midline at x = ")[1].split(",")[0])
        labels_rs = np.asarray(nib.load(out_labels).dataobj)
        prob_rs = np.moveaxis(np.asarray(nib.load(out_prob).dataobj, dtype=np.float32), -1, 0)

    max_diff = float(np.abs(prob_rs - prob_py).max())
    agree = float((labels_rs == labels_py).mean())
    report = {"midline_python": round(midline_py, 3), "midline_rust": midline_rs,
              "max_abs_prob_diff": max_diff, "label_agreement": agree,
              "seconds_python": round(t_py, 1), "seconds_rust": round(t_rs, 1)}
    print(json.dumps(report, indent=2))
    if abs(midline_py - midline_rs) > 1e-2 or max_diff > args.tolerance or agree < 0.99999:
        sys.exit("parity check failed")
    print("parity check passed")


if __name__ == "__main__":
    main()
