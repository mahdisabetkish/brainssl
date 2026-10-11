"""Sliding-window inference over a cropped volume.

Windows always span the full left-right width of the crop, so every window
contains both hemispheres and the bilateral blocks see a voxel's mirror image
at every position. They slide along the other two axes with half overlap, and
overlapping predictions are blended with a Gaussian weight that favours window
centres, where the receptive field is complete.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn


def _starts(size: int, window: int, overlap: float) -> list[int]:
    if size <= window:
        return [0]
    step = max(1, int(window * (1 - overlap)))
    starts = list(range(0, size - window + 1, step))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def _gaussian(shape: tuple[int, ...], sigma_scale: float = 0.125) -> torch.Tensor:
    axes = []
    for n in shape:
        t = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
        axes.append(torch.exp(-0.5 * (t / (sigma_scale * n)) ** 2))
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    w = w / w.max()
    return w.clamp_min(1e-3)


@torch.no_grad()
def sliding_window(model: nn.Module, image: torch.Tensor, window: tuple[int, int, int] = (160, 128, 128),
                   overlap: float = 0.5, batch: int = 2, mirror: bool = True) -> torch.Tensor:
    """Region probabilities (K, X, Y, Z) for one image (C, X, Y, Z) on the model's device.

    With ``mirror`` the prediction is averaged with the prediction on the
    left-right flipped input, flipped back. Inputs smaller than the window are
    zero-padded.
    """
    device = next(model.parameters()).device
    c, *size = image.shape
    pad = [max(0, w - s) for w, s in zip(window, size)]
    if any(pad):
        image = torch.nn.functional.pad(image, (0, pad[2], 0, pad[1], 0, pad[0]))
    _, X, Y, Z = image.shape
    weight = _gaussian(window).to(device)
    out = None
    norm = torch.zeros((X, Y, Z), device=device)
    boxes = list(itertools.product(_starts(X, window[0], overlap), _starts(Y, window[1], overlap),
                                   _starts(Z, window[2], overlap)))
    flips = [False, True] if mirror else [False]
    for k in range(0, len(boxes), batch):
        chunk = boxes[k:k + batch]
        tiles = torch.stack([image[:, x:x + window[0], y:y + window[1], z:z + window[2]]
                             for x, y, z in chunk]).to(device, non_blocking=True)
        probs = 0
        for flip in flips:
            inp = tiles.flip(2) if flip else tiles
            logits = model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            p = torch.sigmoid(logits.float())
            probs = probs + (p.flip(2) if flip else p)
        probs = probs / len(flips)
        if out is None:
            out = torch.zeros((probs.shape[1], X, Y, Z), device=device)
        for (x, y, z), p in zip(chunk, probs):
            out[:, x:x + window[0], y:y + window[1], z:z + window[2]] += p * weight
            norm[x:x + window[0], y:y + window[1], z:z + window[2]] += weight
    out = out / norm.clamp_min(1e-6)
    return out[:, :size[0], :size[1], :size[2]]


def enforce_hierarchy(prob: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binary region masks with ET inside TC inside WT."""
    p = prob.detach().cpu().numpy() if isinstance(prob, torch.Tensor) else prob
    wt = p[0] > threshold
    tc = (p[1] > threshold) & wt
    et = (p[2] > threshold) & tc
    return np.stack([wt, tc, et]).astype(np.uint8)
