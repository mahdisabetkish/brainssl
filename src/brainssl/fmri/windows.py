"""Connectomes of sub-windows of each scan.

A subject's connectome is an estimate from a finite scan, and the scan itself
says how uncertain that estimate is: re-estimating it on overlapping windows
gives samples from (approximately) the subject's own sampling distribution. The
noise-aware classifiers train on these samples instead of on the single
full-scan estimate.

K windows of a fixed fraction of the scan are placed at evenly spaced offsets,
so the first starts at the first volume and the last ends at the last one.
Shorter scans therefore get proportionally shorter, noisier windows, which is
the point: their estimates are noisier, and the classifier sees that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .abide import TimeSeriesSet
from .connectivity import spd_log, sym_to_vec
from .spd_torch import ledoit_wolf_correlation

log = logging.getLogger(__name__)


def window_starts(length: int, n_windows: int, fraction: float, min_length: int = 40) -> tuple[np.ndarray, int]:
    width = min(length, max(min_length, int(fraction * length)))
    return np.linspace(0, length - width, n_windows).round().astype(int), width


def window_log_embeddings(ts: TimeSeriesSet, n_windows: int = 6, fraction: float = 0.7) -> np.ndarray:
    """(subjects, windows, features) log-Euclidean embeddings of window connectomes."""
    out = []
    for i in range(len(ts)):
        x = torch.from_numpy(ts[i]).double()
        starts, width = window_starts(len(x), n_windows, fraction)
        wins = torch.stack([x[s:s + width] for s in starts])
        corr = ledoit_wolf_correlation(wins).numpy()
        out.append(sym_to_vec(spd_log(corr)).astype(np.float32))
    return np.stack(out)


def cached_window_embeddings(ts: TimeSeriesSet, cache_dir: Path, n_windows: int = 6,
                             fraction: float = 0.7) -> np.ndarray:
    path = cache_dir / f"winlogs_{ts.atlas}_{n_windows}_{fraction}.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(ts):
            return emb
    emb = window_log_embeddings(ts, n_windows, fraction)
    np.save(path, emb)
    log.info("window embeddings for %s: %s", ts.atlas, emb.shape)
    return emb
