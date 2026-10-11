"""Fit the deployed connectome classifier on every subject and write it out.

Cross-validation estimates how well the method generalises; the model that is
shipped is refitted once on all 871 subjects with the same settings. For each
atlas the export holds the standardisation (mean and standard deviation of every
feature), the weight vector and the intercept. Arrays are raw little-endian
float32 so that the Rust binary and the browser demo can read them with no
library; ``connectome.json`` records shapes, offsets and everything needed to
recompute the features from a time series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import write_json
from .abide import load_cache
from .atlases import atlas_geometry
from .connectivity import sym_to_vec
from .methods import ConnectomeData, _dual_logreg
from .study import _gram, _standardised_rows, refit_intercept


def fit_atlas(data: ConnectomeData, atlas: str, C: float, vicinal: bool = True) -> dict[str, np.ndarray | float]:
    """The model of the study, fitted on every subject in ``data``.

    With ``vicinal`` the intercept is refitted for the test-time rule exactly as
    in the cross-validation (``study.refit_intercept``), so the shipped model is
    the one that was evaluated.
    """
    full = sym_to_vec(data.logs[atlas])
    mu, sd = full.mean(0), full.std(0) + 1e-8
    n = data.n
    windows = data.windows[atlas] if vicinal else None
    k = 0 if windows is None else windows.shape[1]
    z = _standardised_rows(full, windows, mu, sd)
    gram = _gram(z)
    y = data.labels.astype(np.float64)
    labels = torch.from_numpy(np.concatenate([y, np.repeat(y, k)]))
    weights = torch.from_numpy(np.concatenate([np.ones(n), np.full(n * k, 1.0 / k)])) if k else None
    a, b = _dual_logreg(gram, labels, weights, C)
    if k:
        scores = (gram @ a).numpy()
        b = refit_intercept(scores[:n], scores[n:].reshape(n, k), y)
    # w = Z'a, accumulated in float64 a block of rows at a time
    a = a.numpy()
    w = np.zeros(z.shape[1])
    for s in range(0, len(z), 1024):
        w += z[s:s + 1024].astype(np.float64).T @ a[s:s + 1024]
    return {"mean": mu.astype(np.float32), "std": sd.astype(np.float32), "weight": w.astype(np.float32),
            "bias": float(b)}


def export_connectome(c_by_atlas: dict[str, float], out: Path, raw: Path, processed: Path,
                      n_windows: int = 6, fraction: float = 0.7, vicinal: bool = True) -> Path:
    """Fit one model per atlas on all subjects, loading one atlas at a time."""
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "estimator": "Ledoit-Wolf covariance of z-scored ROI series, rescaled to unit diagonal",
        "embedding": "upper triangle (including diagonal) of log(C), row-major, off-diagonal entries times sqrt(2)",
        "standardise": "z = (x - mean) / std, per feature",
        "score": "p = sigmoid(weight . z + bias)",
        "windows": {"count": n_windows, "fraction": fraction, "min_length": 40,
                    "placement": "evenly spaced starts from 0 to T - width (rounded half to even), "
                                 "width = min(T, max(40, floor(fraction * T)))"},
        "test_time": "p = (p_full + mean of p over the windows) / 2, with the bias refitted for this rule",
        "ensemble": "average p over the atlases supplied",
        "label": {"1": "autism (ASD)", "0": "typically developing control"},
        "atlases": {},
    }
    for atlas, C in c_by_atlas.items():
        data = ConnectomeData.load(raw, processed, (atlas,), windows=vicinal, n_windows=n_windows, fraction=fraction)
        params = fit_atlas(data, atlas, C, vicinal=vicinal)
        del data
        blob = np.concatenate([params["mean"], params["std"], params["weight"]]).astype("<f4")
        blob.tofile(out / f"weights_{atlas}.bin")
        labels = load_cache(processed / "abide", atlas).roi_labels
        geo = atlas_geometry(raw / "abide" / "atlases", atlas, labels)
        d = len(params["weight"])
        meta["atlases"][atlas] = {
            "n_rois": int(len(labels)), "n_features": d, "C": C, "bias": params["bias"],
            "file": f"weights_{atlas}.bin", "layout": {"mean": [0, d], "std": [d, 2 * d], "weight": [2 * d, 3 * d]},
            "roi_labels": labels.tolist(), "centroids_mm": np.round(geo.centroids, 2).tolist(),
        }
    write_json(out / "connectome.json", meta)
    return out
