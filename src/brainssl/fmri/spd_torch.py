"""Batched connectivity estimation and tangent mapping on the GPU.

Self-supervised views are built from random time windows, so the correlation
and its tangent projection have to be recomputed for every view of every
subject in every step. These functions mirror ``connectivity.py`` in torch and
agree with it to float32 precision (see ``tests/test_connectivity.py``).
"""

from __future__ import annotations

import torch


def ledoit_wolf_correlation(x: torch.Tensor, lengths: torch.Tensor | None = None,
                            eps: float = 1e-8) -> torch.Tensor:
    """Shrunk correlation for a batch of (T, R) series, zero-padded along time.

    Implements the Ledoit-Wolf estimator on z-scored data, the same estimator
    as ``sklearn.covariance.ledoit_wolf(..., assume_centered=True)`` after
    standardisation, followed by a rescale to unit diagonal.
    """
    b, t_max, r = x.shape
    if lengths is None:
        lengths = torch.full((b,), t_max, device=x.device)
    mask = (torch.arange(t_max, device=x.device)[None, :] < lengths[:, None]).to(x.dtype)
    n = lengths.to(x.dtype)[:, None]
    mean = (x * mask[..., None]).sum(1) / n
    xc = (x - mean[:, None]) * mask[..., None]
    std = torch.sqrt((xc ** 2).sum(1) / n)
    alive = std > eps
    xs = torch.where(alive[:, None], xc / std.clamp_min(eps)[:, None], torch.zeros_like(xc))

    s = xs.transpose(1, 2) @ xs / n[..., None]                          # (b, r, r)
    mu = torch.diagonal(s, dim1=1, dim2=2).sum(-1) / r                 # (b,)
    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    delta = ((s - mu[:, None, None] * eye) ** 2).sum((1, 2)) / r
    # sum_t ||x_t x_t^T - S||_F^2 / T^2, expanded to avoid the (b, t, r, r) tensor.
    sq = (xs ** 2).sum(-1)                                             # (b, t)
    beta = ((sq ** 2).sum(1) / n[:, 0] - (s ** 2).sum((1, 2))) / n[:, 0] / r
    beta = torch.minimum(beta, delta)
    shrink = torch.where(delta > 0, beta / delta.clamp_min(eps), torch.zeros_like(delta))
    cov = shrink[:, None, None] * mu[:, None, None] * eye + (1 - shrink[:, None, None]) * s
    d = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(eps))
    return cov / (d[:, :, None] * d[:, None, :])


def _eig_apply(mats: torch.Tensor, fn) -> torch.Tensor:
    w, v = torch.linalg.eigh(mats)
    return (v * fn(w)[..., None, :]) @ v.transpose(-1, -2)


def spd_log(mats: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return _eig_apply(mats, lambda w: torch.log(w.clamp_min(floor)))


def tangent(mats: torch.Tensor, whitener: torch.Tensor) -> torch.Tensor:
    """log(W C W) for a whitening matrix W = M^{-1/2} fitted on training data."""
    return spd_log(whitener @ mats @ whitener)
