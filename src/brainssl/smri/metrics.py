"""Dice and 95th-percentile Hausdorff distance under the BraTS conventions.

When a region is absent from both the reference and the prediction, Dice is 1
and HD95 is 0. When it is absent from exactly one of them, Dice is 0 and HD95
takes the penalty value 373.13 mm, the diagonal of the 240 x 240 x 155 grid,
which is what the BraTS evaluation uses.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

HD_PENALTY = 373.13


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    total = p.sum() + r.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, r).sum() / total)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() and not r.any():
        return 0.0
    if not p.any() or not r.any():
        return HD_PENALTY
    # Distances are only needed near the two objects; crop to their joint box.
    idx = np.argwhere(p | r)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = idx.max(0) + 3
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, r = p[box], r[box]
    sp, sr = _surface(p), _surface(r)
    dist_to_r = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dist_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    # Larger of the two directed 95th percentiles, as in MONAI and medpy.
    return float(max(np.percentile(dist_to_r[sp], 95), np.percentile(dist_to_p[sr], 95)))


def region_metrics(pred_regions: np.ndarray, ref_regions: np.ndarray,
                   names: tuple[str, ...] = ("WT", "TC", "ET"), distances: bool = True) -> dict[str, float]:
    """Dice (and HD95 unless ``distances`` is False, which is ~5x faster) per region."""
    out = {}
    for k, name in enumerate(names):
        out[f"dice_{name}"] = dice(pred_regions[k], ref_regions[k])
        if distances:
            out[f"hd95_{name}"] = hd95(pred_regions[k], ref_regions[k])
    out["dice_mean"] = float(np.mean([out[f"dice_{n}"] for n in names]))
    if distances:
        out["hd95_mean"] = float(np.mean([out[f"hd95_{n}"] for n in names]))
    return out
