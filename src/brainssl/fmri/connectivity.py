"""Functional connectivity as points on the manifold of SPD matrices.

A shrunk correlation matrix is symmetric positive definite, and the set of such
matrices is not a vector space: averaging or subtracting them entrywise ignores
the curvature, and a classifier fitted to raw coefficients sees distances that
depend on where in the cone a subject happens to sit. The usual fix is to map
every matrix to the tangent space at a reference point M,

    T(C) = log(M^{-1/2} C M^{-1/2}),

where the affine-invariant geometry becomes approximately Euclidean, and to
choose M as the Frechet mean of the training subjects. Vectorising the upper
triangle with off-diagonal entries scaled by sqrt(2) makes the Euclidean norm of
the vector equal to the Frobenius norm of the matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf


def _eig_apply(mats: np.ndarray, fn) -> np.ndarray:
    """Apply a scalar function to the eigenvalues of symmetric matrices.

    The decomposition goes through torch: on the machine used here its batched
    CPU eigh is several times faster than numpy's for 100-400 dimensional
    matrices, and the results agree to double precision.
    """
    w, v = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(mats, dtype=np.float64)))
    w, v = w.numpy(), v.numpy()
    return (v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def spd_sqrt(mats: np.ndarray) -> np.ndarray:
    return _eig_apply(mats, np.sqrt)


def spd_invsqrt(mats: np.ndarray) -> np.ndarray:
    return _eig_apply(mats, lambda w: 1.0 / np.sqrt(w))


def spd_log(mats: np.ndarray) -> np.ndarray:
    return _eig_apply(mats, np.log)


def sym_exp(mats: np.ndarray) -> np.ndarray:
    return _eig_apply(mats, np.exp)


def shrunk_correlation(ts: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance of z-scored time series, i.e. a shrunk correlation.

    ROIs whose signal is constant (they fall outside the subject's field of
    view) get zero rows before shrinkage, so they end up uncorrelated with every
    other region and the matrix stays positive definite.
    """
    x = ts.astype(np.float64)
    x = x - x.mean(axis=0)
    std = x.std(axis=0)
    dead = std < 1e-8
    x[:, ~dead] /= std[~dead]
    x[:, dead] = 0.0
    cov, _ = ledoit_wolf(x, assume_centered=True)
    # Shrinkage puts the same value on every diagonal entry of a dead ROI; rescale
    # to unit diagonal so all subjects live on correlation-like matrices.
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def batch_correlations(series: list[np.ndarray]) -> np.ndarray:
    return np.stack([shrunk_correlation(ts) for ts in series])


def frechet_mean(mats: np.ndarray, max_iter: int = 10, tol: float = 1e-6,
                 logs: np.ndarray | None = None) -> np.ndarray:
    """Affine-invariant Frechet (Karcher) mean by Riemannian gradient descent.

    ``logs`` may hold precomputed matrix logarithms of ``mats``; they give the
    log-Euclidean mean, which is close to the answer and a stable start.
    """
    if logs is None:
        logs = spd_log(mats)
    mean = sym_exp(logs.mean(axis=0))
    for _ in range(max_iter):
        root, inv_root = spd_sqrt(mean), spd_invsqrt(mean)
        step = spd_log(inv_root @ mats @ inv_root).mean(axis=0)
        mean = root @ sym_exp(step) @ root
        if np.linalg.norm(step) / np.sqrt(len(step)) < tol:
            break
    return 0.5 * (mean + mean.T)


def log_euclidean_mean(logs: np.ndarray) -> np.ndarray:
    """exp of the average matrix logarithm (Arsigny et al., 2007)."""
    return sym_exp(logs.mean(axis=0))


def sym_to_vec(mats: np.ndarray, discard_diagonal: bool = False) -> np.ndarray:
    """Upper triangle with off-diagonal entries scaled by sqrt(2)."""
    n = mats.shape[-1]
    k = 1 if discard_diagonal else 0
    iu = np.triu_indices(n, k=k)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return mats[..., iu[0], iu[1]] * scale


def vec_to_sym(vec: np.ndarray, n: int) -> np.ndarray:
    iu = np.triu_indices(n)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.zeros(vec.shape[:-1] + (n, n), dtype=vec.dtype)
    out[..., iu[0], iu[1]] = vec / scale
    out[..., iu[1], iu[0]] = vec / scale
    return out


class TangentSpace:
    """Fit a reference point on training matrices and project any matrix.

    ``reference="frechet"`` is the affine-invariant mean used by Varoquaux et
    al. (2010) and in nilearn's ``ConnectivityMeasure(kind="tangent")``;
    ``"log-euclidean"`` and ``"identity"`` are cheaper alternatives.
    """

    def __init__(self, reference: str = "frechet") -> None:
        self.reference = reference
        self.mean_: np.ndarray | None = None
        self._whiten: np.ndarray | None = None

    def fit(self, mats: np.ndarray, logs: np.ndarray | None = None) -> TangentSpace:
        if self.reference == "frechet":
            mean = frechet_mean(mats, logs=logs)
        elif self.reference == "log-euclidean":
            mean = log_euclidean_mean(spd_log(mats) if logs is None else logs)
        elif self.reference == "identity":
            mean = np.eye(mats.shape[-1])
        else:
            raise ValueError(self.reference)
        self.mean_ = mean
        self._whiten = spd_invsqrt(mean)
        return self

    def transform(self, mats: np.ndarray, vectorize: bool = True) -> np.ndarray:
        if self._whiten is None:
            raise RuntimeError("call fit first")
        t = spd_log(self._whiten @ mats @ self._whiten)
        return sym_to_vec(t) if vectorize else t

    def fit_transform(self, mats: np.ndarray, vectorize: bool = True) -> np.ndarray:
        return self.fit(mats).transform(mats, vectorize)


def fisher_z(corr: np.ndarray) -> np.ndarray:
    """Fisher-transformed off-diagonal correlations, vectorised."""
    n = corr.shape[-1]
    iu = np.triu_indices(n, k=1)
    r = np.clip(corr[..., iu[0], iu[1]], -0.999999, 0.999999)
    return np.arctanh(r)
