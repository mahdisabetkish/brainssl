import numpy as np

from brainssl.fmri.connectivity import shrunk_correlation, spd_log, sym_to_vec
from brainssl.fmri.evaluate import Fold
from brainssl.fmri.methods import ConnectomeData
from brainssl.fmri.study import refit_intercept, run_atlas_fold, run_study
from brainssl.fmri.windows import window_starts


def test_window_starts_cover_the_scan():
    starts, width = window_starts(196, 6, 0.7)
    assert width == 137
    assert starts[0] == 0 and starts[-1] + width == 196
    assert list(starts) == [0, 12, 24, 35, 47, 59]          # pinned in the Rust tests too
    starts, width = window_starts(50, 6, 0.7)
    assert width == 40                                       # the minimum window length


def test_refit_intercept_matches_class_balance(rng):
    s = rng.normal(size=400)
    y = (rng.uniform(size=400) < 0.3).astype(float)          # scores carry no signal
    b = refit_intercept(s * 0.0, None, y)
    assert abs(1 / (1 + np.exp(-b)) - y.mean()) < 1e-3


def _toy_data(rng, n=60, r=8, k=3):
    labels = np.repeat([0, 1], n // 2)
    sites = np.array(["a", "b"] * (n // 2))
    corr, logs, wins = [], [], []
    for i in range(n):
        mix = rng.normal(size=(r, r)) + labels[i] * np.eye(r)
        ts = rng.normal(size=(80, r)) @ mix
        c = shrunk_correlation(ts)
        corr.append(c)
        logs.append(spd_log(c))
        wins.append([sym_to_vec(spd_log(shrunk_correlation(ts[s:s + 56]))) for s in (0, 12, 24)][:k])
    return ConnectomeData(labels, sites, ("toy",), {"toy": np.stack(corr)}, {"toy": np.stack(logs)},
                          {"toy": np.stack(wins).astype(np.float32)})


def test_study_fold_produces_probabilities_for_every_model(rng):
    data = _toy_data(rng)
    idx = rng.permutation(data.n)
    fold = Fold(0, 0, np.sort(idx[:45]), np.sort(idx[45:]))
    grids = {m: (1e-2, 1e-1) for m in ("tangent", "log", "corr", "vicinal")}
    out = run_atlas_fold(data, "toy", fold, grids)
    assert set(out) == {"tangent", "log", "corr", "vicinal"}
    for by_c in out.values():
        for p in by_c.values():
            assert p.shape == (15,) and np.all((p >= 0) & (p <= 1))


def test_study_completes_a_split_with_new_values_of_c(rng, tmp_path):
    data = _toy_data(rng)
    idx = rng.permutation(data.n)
    folds = [Fold(0, 0, np.sort(idx[:45]), np.sort(idx[45:])), Fold(0, 1, np.sort(idx[15:]), np.sort(idx[:15]))]
    run_study(data, ("toy",), folds, tmp_path, "kfold", {"log": (1e-1,), "vicinal": (1e-1,)})
    first = np.load(tmp_path / "kfold" / "log" / "toy" / "C0.1.npy")
    # a wider grid later: only the new pair is fitted, the old one is kept as saved
    run_study(data, ("toy",), folds, tmp_path, "kfold", {"log": (1e-2, 1e-1), "vicinal": (1e-1,)})
    assert np.array_equal(np.load(tmp_path / "kfold" / "log" / "toy" / "C0.1.npy"), first, equal_nan=True)
    saved = np.load(tmp_path / "kfold" / "partial" / "toy" / f"{folds[0].tag}.npz")
    assert set(saved.files) == {"log|0.1", "log|0.01", "vicinal|0.1"}
    tested = np.concatenate([f.test for f in folds])
    assert np.isfinite(np.load(tmp_path / "kfold" / "log" / "toy" / "C0.01.npy")[0, tested]).all()
