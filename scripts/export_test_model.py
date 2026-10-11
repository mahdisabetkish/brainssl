"""Export a small segmentation network with random weights, for CI parity tests.

The Rust binary has to reproduce the Python pipeline for any weights, so the
check does not need the trained model (which is not stored in git).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brainssl.export.onnx_export import export_segmentation
from brainssl.smri.model import UNet3D, UNetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    channels = (4, 8, 16)
    model = UNet3D(UNetConfig(channels=channels, bilateral=True, bilateral_levels=(1, 2)))
    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.out / "weights.pt"
    torch.save(model.state_dict(), weights)
    export_segmentation(weights, args.out, bilateral=True, channels=channels, window=(32, 32, 32))
    weights.unlink()


if __name__ == "__main__":
    main()
