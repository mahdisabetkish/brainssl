"""ONNX export of trained models, with the metadata a runtime needs.

The segmentation network is exported for one sliding window: input ``image``
(N, 4, 160, 128, 128), output ``logits`` (N, 3, 160, 128, 128). Everything around
it (cropping about the midline, normalisation, window placement, blending,
thresholds) is described in ``segmentation.json`` so that the Rust binary and
the browser demo reproduce the Python pipeline from the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..smri.model import UNet3D, UNetConfig
from ..smri.preprocess import CHANNELS, LR_WIDTH, REGIONS
from ..utils import write_json


def export_segmentation(weights: Path, out_dir: Path, bilateral: bool,
                        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
                        window: tuple[int, int, int] = (LR_WIDTH, 128, 128), opset: int = 17,
                        check: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = UNet3D(UNetConfig(channels=channels, bilateral=bilateral, deep_supervision=False))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith(("extra.", "aux."))}
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.randn(1, len(CHANNELS), *window)
    path = out_dir / "segmentation.onnx"
    torch.onnx.export(model, (dummy,), str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    write_json(out_dir / "segmentation.json", {
        "input": "image", "output": "logits", "window": list(window), "overlap": 0.5,
        "channels": list(CHANNELS), "regions": list(REGIONS), "threshold": 0.5,
        "lr_width": window[0], "crop_multiple": 16, "crop_margin": 2, "mirror_tta": True,
        "gaussian_sigma_scale": 0.125, "bilateral": bilateral, "unet_channels": list(channels),
        "labels": {"1": "oedema", "2": "non-enhancing tumour core", "3": "enhancing tumour"},
    })
    if check:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).normal(size=(1, len(CHANNELS), *window)).astype(np.float32)
        ref = model(torch.from_numpy(x)).detach().numpy()
        got = session.run(["logits"], {"image": x})[0]
        err = float(np.abs(ref - got).max())
        if err > 1e-3:
            raise RuntimeError(f"ONNX output differs from PyTorch by {err:.2e}")
    return path
