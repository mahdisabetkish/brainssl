import numpy as np
import pytest
import torch
from sklearn.covariance import ledoit_wolf

from brainssl.fmri.connectivity import (
    TangentSpace,
    fisher_z,
    frechet_mean,
    shrunk_correlation,
    spd_log,
    sym_exp,
    sym_to_vec,
    vec_to_sym,
)
from brainssl.fmri.spd_torch import ledoit_wolf_correlation, tangent


def _series(rng, t=120, r=20):
    mix = rng.normal(size=(r, r))
    return rng.normal(size=(t, r)) @ mix


def test_shrunk_correlation_is_ledoit_wolf_on_zscored_data(rng):
    x = _series(rng)
    z = (x - x.mean(0)) / x.std(0)
    cov, _ = ledoit_wolf(z, assume_centered=True)
    d = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(shrunk_correlation(x), cov / np.outer(d, d), atol=1e-10)


def test_dead_roi_stays_positive_definite(rng):
    x = _series(rng)
    x[:, 3] = 0.0
    c = shrunk_correlation(x)
    assert np.linalg.eigvalsh(c).min() > 0
    np.testing.assert_allclose(np.diag(c), 1.0)
    assert np.allclose(np.delete(c[3], 3), 0.0)


def test_torch_estimator_matches_numpy_with_padding(rng):
    a, b = _series(rng, 120), _series(rng, 90)
    batch = np.zeros((2, 120, a.shape[1]))
    batch[0], batch[1, :90] = a, b
    out = ledoit_wolf_correlation(torch.from_numpy(batch), torch.tensor([120, 90])).numpy()
    np.testing.assert_allclose(out[0], shrunk_correlation(a), atol=1e-10)
    np.testing.assert_allclose(out[1], shrunk_correlation(b), atol=1e-10)


def test_log_and_exp_are_inverse(rng):
    c = shrunk_correlation(_series(rng))
    np.testing.assert_allclose(sym_exp(spd_log(c)), c, atol=1e-10)


def test_frechet_mean_of_commuting_matrices_is_geometric_mean():
    d = np.diag([1.0, 4.0, 9.0])
    e = np.diag([4.0, 1.0, 1.0])
    m = frechet_mean(np.stack([d, e]), max_iter=50, tol=1e-12)
    np.testing.assert_allclose(m, np.diag([2.0, 2.0, 3.0]), atol=1e-8)


def test_frechet_mean_matches_nilearn(rng):
    connectivity = pytest.importorskip("nilearn.connectome.connectivity_matrices")
    mats = np.stack([shrunk_correlation(_series(rng)) for _ in range(6)])
    ours = TangentSpace("frechet").fit(mats).mean_
    ref = connectivity._geometric_mean(list(mats), max_iter=50, tol=1e-9)
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_tangent_vectors_are_isometric(rng):
    mats = np.stack([shrunk_correlation(_series(rng)) for _ in range(4)])
    ts = TangentSpace("frechet").fit(mats)
    t = ts.transform(mats, vectorize=False)
    v = ts.transform(mats)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.linalg.norm(t, axis=(1, 2)), rtol=1e-10)
    np.testing.assert_allclose(vec_to_sym(sym_to_vec(t), t.shape[-1]), t, atol=1e-12)


def test_torch_tangent_matches_numpy(rng):
    mats = np.stack([shrunk_correlation(_series(rng)) for _ in range(3)])
    ts = TangentSpace("log-euclidean").fit(mats)
    w = torch.from_numpy(ts._whiten)
    out = tangent(torch.from_numpy(mats), w).numpy()
    np.testing.assert_allclose(out, ts.transform(mats, vectorize=False), atol=1e-8)


def test_fisher_z_shape(rng):
    c = shrunk_correlation(_series(rng, r=10))
    assert fisher_z(c).shape == (45,)
