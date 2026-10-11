"""Indexing, splitting and loading of the MSD brain tumour volumes.

Every volume is 240 x 240 x 155 at 1 mm with four co-registered contrasts
(FLAIR, T1, T1 with gadolinium, T2), skull-stripped and rigidly aligned to the
SRI24 template by the BraTS organisers. Axis 0 runs left to right.

Two facts about the data shape the pipeline. Background voxels are exactly zero,
so the brain mask is free. And alignment to a symmetric template puts the
mid-sagittal plane close to column 119.5 in every scan, which is what lets a
flip of axis 0 stand in for reflection across the midline. The plane is still
estimated per scan, by locating the peak of the cross-correlation between the
brain mask and its mirror image, and each scan is cropped to a 160-voxel window
centred on it.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHANNELS = ("FLAIR", "T1", "T1Gd", "T2")
LR_WIDTH = 160
LABELS = {0: "background", 1: "oedema", 2: "non-enhancing tumour", 3: "enhancing tumour"}
REGIONS = ("WT", "TC", "ET")   # whole tumour {1,2,3}, tumour core {2,3}, enhancing tumour {3}


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    label: Path | None
    source: str   # "train" (labelled release) or "test" (unlabelled release)


def list_cases(task_dir: Path) -> list[Case]:
    cases = []
    for path in sorted((task_dir / "imagesTr").glob("BRATS_*.nii.gz")):
        cases.append(Case(path.name[:-7], path, task_dir / "labelsTr" / path.name, "train"))
    for path in sorted((task_dir / "imagesTs").glob("BRATS_*.nii.gz")):
        cases.append(Case(path.name[:-7], path, None, "test"))
    return cases


def read_volume(path: Path) -> np.ndarray:
    """(4, 240, 240, 155) float32, channels first."""
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return np.ascontiguousarray(np.moveaxis(data, -1, 0))


def read_volume_cropped(path: Path, box: tuple[slice, ...]) -> np.ndarray:
    """Like ``_take(read_volume(path), box)`` but crops before the channels-first
    copy, which roughly halves the peak memory of a data-loading worker."""
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)       # (X, Y, Z, C)
    return _take(np.moveaxis(data, -1, 0), box)


def read_label(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=np.uint8)


def estimate_midline(mask: np.ndarray, max_shift: int = 30) -> float:
    """Reflection centre c along axis 0, so that x -> 2c - x maps the mask onto itself.

    Reflecting about c is a flip followed by a shift of k = 2c - (n - 1) voxels.
    The circular cross-correlation of the mask with its flipped copy, summed over
    the other two axes, gives the overlap for every k at once; a parabola through
    the peak and its neighbours gives sub-voxel precision.
    """
    m = mask.astype(np.float32)
    n = m.shape[0]
    size = 2 * n
    spectrum = np.fft.rfft(m, n=size, axis=0) * np.conj(np.fft.rfft(m[::-1], n=size, axis=0))
    overlap = np.fft.irfft(spectrum, n=size, axis=0).sum(axis=(1, 2))
    shifts = np.arange(size)
    shifts[shifts >= n] -= size
    keep = np.abs(shifts) <= max_shift
    overlap, shifts = overlap[keep], shifts[keep]
    order = np.argsort(shifts)
    overlap, shifts = overlap[order], shifts[order]
    i = int(np.argmax(overlap))
    offset = 0.0
    if 0 < i < len(overlap) - 1:
        a, b, c = overlap[i - 1], overlap[i], overlap[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            offset = 0.5 * (a - c) / denom
    return (n - 1 + shifts[i] + offset) / 2.0


def summarize(case: Case, raw_root: Path | None = None) -> dict:
    """Per-case statistics. Paths are stored relative to ``raw_root`` in POSIX
    form, so an index built on one machine works on any other."""
    image = read_volume(case.image)
    mask = (image > 0).any(axis=0)
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    def rel(path: Path) -> str:
        return (path.resolve().relative_to(raw_root.resolve()) if raw_root else path).as_posix()

    row: dict = {
        "case_id": case.case_id, "source": case.source,
        "image": rel(case.image), "label": rel(case.label) if case.label else "",
        "midline": estimate_midline(mask),
        "x0": int(lo[0]), "x1": int(hi[0]), "y0": int(lo[1]), "y1": int(hi[1]),
        "z0": int(lo[2]), "z1": int(hi[2]), "brain_voxels": int(mask.sum()),
        "nx": int(mask.shape[0]), "ny": int(mask.shape[1]), "nz": int(mask.shape[2]),
    }
    for c, name in enumerate(CHANNELS):
        values = image[c][mask]
        row[f"{name}_mean"] = float(values.mean())
        row[f"{name}_std"] = float(values.std())
    if case.label is not None:
        label = read_label(case.label)
        for k in (1, 2, 3):
            row[f"n_label{k}"] = int((label == k).sum())
    return row


def build_index(raw_root: Path, out_csv: Path, workers: int = 8) -> pd.DataFrame:
    """Index every case under ``raw_root/Task01_BrainTumour``."""
    cases = list_cases(raw_root / "Task01_BrainTumour")
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(summarize, cases, [raw_root] * len(cases), chunksize=4))
    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("indexed %d cases (%d labelled)", len(df), int((df.source == "train").sum()))
    return df


def make_splits(index: pd.DataFrame, seed: int = 0, test_frac: float = 0.2, val_frac: float = 0.16,
                fractions: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)) -> dict:
    """Fixed test, validation and nested label-budget subsets of the labelled cases.

    Cases are stratified on tumour volume tertile and on whether any enhancing
    tumour is present, so all subsets see the same mix of small, large,
    enhancing and non-enhancing tumours. Label budgets are prefixes of one
    random order of the training pool, so the 5% cases are inside the 10% cases
    and so on, which removes one source of noise from the label-efficiency curve.
    The unlabelled release plus the training pool form the pretraining set;
    validation and test images are never seen during pretraining.
    """
    rng = np.random.default_rng(seed)
    labelled = index[index.source == "train"].copy()
    volume = labelled[["n_label1", "n_label2", "n_label3"]].sum(axis=1)
    tertile = pd.qcut(volume, 3, labels=False)
    has_et = (labelled["n_label3"] > 0).astype(int)
    labelled["stratum"] = tertile.astype(str) + "_" + has_et.astype(str)

    test, val, pool = [], [], []
    for _, group in labelled.groupby("stratum"):
        ids = group["case_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_test = int(round(test_frac * len(ids)))
        n_val = int(round(val_frac * len(ids)))
        test += ids[:n_test].tolist()
        val += ids[n_test:n_test + n_val].tolist()
        pool += ids[n_test + n_val:].tolist()
    pool = np.array(sorted(pool))
    rng.shuffle(pool)
    budgets = {f"{int(round(100 * f))}pct": sorted(pool[:max(1, int(round(f * len(pool))))].tolist())
               for f in fractions}
    unlabelled = sorted(index[index.source == "test"]["case_id"].tolist())
    return {
        "seed": seed,
        "test": sorted(test),
        "val": sorted(val),
        "train_pool": sorted(pool.tolist()),
        "budgets": budgets,
        "pretrain": sorted(pool.tolist() + unlabelled),
    }


def _grid_shape(row: pd.Series) -> tuple[int, int, int]:
    if "nx" in row and not pd.isna(row["nx"]):
        return int(row["nx"]), int(row["ny"]), int(row["nz"])
    return (240, 240, 155)


def crop_box(row: pd.Series, shape: tuple[int, int, int] = (240, 240, 155),
             lr_width: int = LR_WIDTH, multiple: int = 16, margin: int = 2) -> tuple[slice, slice, slice]:
    """Crop that is symmetric about the estimated midline along axis 0.

    A window [x0, x0 + w) is mirrored onto itself by a flip exactly when
    x0 + (w - 1) / 2 equals the reflection centre, so x0 is rounded from
    c - (w - 1) / 2 and the residual misalignment is at most half a voxel.
    Axes 1 and 2 keep the brain's bounding box plus a margin, grown to a
    multiple of 16 so the U-Net's downsampling divides evenly.
    """
    x0 = int(round(row["midline"] - (lr_width - 1) / 2))
    out = [slice(x0, x0 + lr_width)]
    for _axis, (lo, hi) in enumerate(((row["y0"], row["y1"]), (row["z0"], row["z1"])), start=1):
        lo, hi = int(lo) - margin, int(hi) + margin
        size = -(-(hi - lo) // multiple) * multiple
        centre = (lo + hi) // 2
        start = centre - size // 2
        out.append(slice(start, start + size))
    return tuple(out)  # type: ignore[return-value]


def _take(array: np.ndarray, box: tuple[slice, ...], fill: float = 0.0) -> np.ndarray:
    """Crop with zero padding where the box leaves the volume."""
    lead = array.shape[: array.ndim - 3]
    shape = lead + tuple(s.stop - s.start for s in box)
    out = np.full(shape, fill, dtype=array.dtype)
    src, dst = [], []
    for s, n in zip(box, array.shape[-3:]):
        a, b = max(s.start, 0), min(s.stop, n)
        src.append(slice(a, b))
        dst.append(slice(a - s.start, b - s.start))
    out[(...,) + tuple(dst)] = array[(...,) + tuple(src)]
    return out


def normalize(image: np.ndarray) -> np.ndarray:
    """Z-score each contrast over brain voxels; background stays at zero."""
    mask = (image > 0).any(axis=0)
    out = np.zeros_like(image)
    for c in range(image.shape[0]):
        values = image[c][mask]
        out[c][mask] = (values - values.mean()) / (values.std() + 1e-6)
    return out


def load_case(row: pd.Series, with_label: bool = True, lr_width: int = LR_WIDTH,
              raw_root: Path | str = "data/raw") -> tuple[np.ndarray, np.ndarray | None, tuple[slice, ...]]:
    """Cropped, normalised image (4, X, Y, Z), its label if any, and the crop box."""
    root = Path(raw_root)
    box = crop_box(row, shape=_grid_shape(row), lr_width=lr_width)
    image = normalize(read_volume_cropped(root / row["image"], box))
    label = None
    if with_label and isinstance(row.get("label"), str) and row["label"]:
        label = _take(read_label(root / row["label"]), box)
    return image, label, box


def uncrop(prediction: np.ndarray, box: tuple[slice, ...],
           shape: tuple[int, int, int] = (240, 240, 155)) -> np.ndarray:
    """Place a cropped prediction back into the original grid."""
    out = np.zeros(prediction.shape[: prediction.ndim - 3] + shape, dtype=prediction.dtype)
    src, dst = [], []
    for s, n in zip(box, shape):
        a, b = max(s.start, 0), min(s.stop, n)
        dst.append(slice(a, b))
        src.append(slice(a - s.start, b - s.start))
    out[(...,) + tuple(dst)] = prediction[(...,) + tuple(src)]
    return out


def regions_from_labels(label: np.ndarray) -> np.ndarray:
    """(3, ...) uint8 masks for whole tumour, tumour core and enhancing tumour."""
    return np.stack([label > 0, (label == 2) | (label == 3), label == 3]).astype(np.uint8)


def labels_from_regions(regions: np.ndarray) -> np.ndarray:
    """Inverse of ``regions_from_labels`` for nested binary masks."""
    wt, tc, et = regions.astype(bool)
    label = np.zeros(wt.shape, dtype=np.uint8)
    label[wt] = 1
    label[tc & wt] = 2
    label[et & tc & wt] = 3
    return label


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1), newline="\n")
