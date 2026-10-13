"""The reported fMRI study: every linear model, every atlas, every split.

For each atlas and each split the features are computed once and shared by all
models and regularisation strengths:

* ``tangent``  affine-invariant tangent space at the training Frechet mean
* ``log``      log-Euclidean embedding vec(log C)
* ``corr``     Fisher-transformed correlations
* ``vicinal``  log-Euclidean, trained on each scan's window connectomes, with
               test-time averaging over the windows and the intercept refitted
               on the training full scans (see docs/methods/fmri.md)

Features are standardised with statistics of the training subjects of the
split. Baselines are fitted at five values of C spanning two orders of
magnitude, and the report credits each with its best C on these very splits,
which favours the baselines; the noise-aware model is fitted at the two values
of its development grid, and its headline result uses the one fixed in advance
from development runs on other random splits (assets/fmri_choices.json).

Each split is saved as soon as it is done, and a split file that lacks some
(model, C) pair is completed rather than recomputed, so the grids can be
extended after the fact at the cost of the missing fits only.

Out-of-fold probabilities are written to
``<out>/<scheme>/<model>/<atlas>/C<C>.npy`` as (repeats, subjects) arrays, so the
atlas ensembles and every comparison are formed afterwards without refitting.

    brainssl fmri study --scheme kfold --atlases cc200 cc400 aal ho dosenbach160
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from ..utils import write_json
from .abide import ATLASES
from .connectivity import TangentSpace, fisher_z, sym_to_vec
from .evaluate import Fold, kfold_splits, site_splits
from .methods import ConnectomeData, _dual_logreg

log = logging.getLogger(__name__)

BASELINE_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
VICINAL_GRID = (1e-4, 3e-4)          # the development grid of scripts/choose_fmri_c.py


def default_grids(models: set[str] | list[str]) -> dict[str, tuple[float, ...]]:
    return {m: VICINAL_GRID if m == "vicinal" else BASELINE_GRID for m in models}


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def refit_intercept(s_full: np.ndarray, s_win: np.ndarray | None, y: np.ndarray) -> float:
    """Intercept maximising the training likelihood of the test-time prediction rule."""
    def nll(b: float) -> float:
        p = _sigmoid(s_full + b) if s_win is None else 0.5 * (_sigmoid(s_full + b) + _sigmoid(s_win + b).mean(1))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(minimize_scalar(nll, bounds=(-20, 20), method="bounded").x)


def _gram(z: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    return (t @ t.T).double()


def _standardised_rows(full: np.ndarray, windows: np.ndarray | None, mu: np.ndarray, sd: np.ndarray,
                       block: int = 512) -> np.ndarray:
    """Full-scan rows, then every window row, standardised and stored as float32.

    Written block by block: for the 392-region atlas the float64 intermediate
    would need about 7 GB.
    """
    n, d = full.shape
    rows = windows.reshape(-1, d) if windows is not None else full[:0]
    z = np.empty((n + len(rows), d), dtype=np.float32)
    z[:n] = (full - mu) / sd
    for s in range(0, len(rows), block):
        z[n + s:n + s + block] = (rows[s:s + block] - mu) / sd
    return z


def _fit_predict_plain(gram: torch.Tensor, tr: np.ndarray, te: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    a, b = _dual_logreg(gram[tr][:, tr], torch.from_numpy(y[tr].astype(np.float64)), None, C)
    return _sigmoid((gram[te][:, tr] @ a).numpy() + float(b))


def run_atlas_fold(data: ConnectomeData, atlas: str, fold: Fold,
                   grids: dict[str, tuple[float, ...]]) -> dict[str, dict[float, np.ndarray]]:
    """Held-out probabilities of every model in ``grids`` at each of its values of C."""
    y = data.labels.astype(np.float64)
    tr, te = fold.train, fold.test
    models = {m for m, cs in grids.items() if cs}
    out: dict[str, dict[float, np.ndarray]] = {}

    def standardise(x: np.ndarray) -> np.ndarray:
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
        return (x - mu) / sd

    if "tangent" in models:
        ts = TangentSpace("frechet").fit(data.corr[atlas][tr], logs=data.logs[atlas][tr])
        gram = _gram(standardise(ts.transform(data.corr[atlas])))
        out["tangent"] = {C: _fit_predict_plain(gram, tr, te, y, C) for C in grids["tangent"]}
    if "corr" in models:
        gram = _gram(standardise(fisher_z(data.corr[atlas])))
        out["corr"] = {C: _fit_predict_plain(gram, tr, te, y, C) for C in grids["corr"]}
    if "log" in models or "vicinal" in models:
        full = sym_to_vec(data.logs[atlas])
        mu, sd = full[tr].mean(0), full[tr].std(0) + 1e-8
        n = data.n
        windows = data.windows[atlas] if "vicinal" in models else None
        k = 0 if windows is None else windows.shape[1]
        gram = _gram(_standardised_rows(full, windows, mu, sd))
        if "log" in models:
            out["log"] = {C: _fit_predict_plain(gram, tr, te, y, C) for C in grids["log"]}
        if "vicinal" in models:
            win = lambda idx: (n + idx[:, None] * k + np.arange(k)[None]).ravel()  # noqa: E731
            rows = np.concatenate([tr, win(tr)])
            labels = torch.from_numpy(np.concatenate([y[tr], np.repeat(y[tr], k)]))
            weights = torch.from_numpy(np.concatenate([np.ones(len(tr)), np.full(len(tr) * k, 1.0 / k)]))
            sub = gram[rows][:, rows]
            out["vicinal"] = {}
            for C in grids["vicinal"]:
                a, _ = _dual_logreg(sub, labels, weights, C)

                def score(idx, a=a):
                    return (gram[idx][:, rows] @ a).numpy()
                b = refit_intercept(score(tr), score(win(tr)).reshape(len(tr), k), y[tr])
                s_te, s_tw = score(te), score(win(te)).reshape(len(te), k)
                out["vicinal"][C] = 0.5 * (_sigmoid(s_te + b) + _sigmoid(s_tw + b).mean(1))
    return out


def run_study(data: ConnectomeData, atlases: tuple[str, ...], folds: list[Fold], out: Path, scheme: str,
              grids: dict[str, tuple[float, ...]]) -> None:
    repeats = 1 + max(f.repeat for f in folds)
    root = out / scheme
    for atlas in atlases:
        partial = root / "partial" / atlas
        partial.mkdir(parents=True, exist_ok=True)
        t0, computed = time.time(), 0
        for i, fold in enumerate(folds):
            path = partial / f"{fold.tag}.npz"
            saved = dict(np.load(path)) if path.exists() else {}
            missing = {m: tuple(C for C in cs if f"{m}|{C}" not in saved) for m, cs in grids.items()}
            missing = {m: cs for m, cs in missing.items() if cs}
            if not missing:
                continue
            res = run_atlas_fold(data, atlas, fold, missing)
            saved.update({f"{m}|{C}": p for m, by_c in res.items() for C, p in by_c.items()})
            # written under a temporary name first, so an interrupted run never leaves a truncated split
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **saved)
            os.replace(tmp, path)
            computed += 1
            if computed % 10 == 0 or i == len(folds) - 1:
                log.info("%s %s: %d/%d splits (%.0fs)", scheme, atlas, i + 1, len(folds), time.time() - t0)
        # out-of-fold arrays of the requested grid, one (repeats, subjects) array per model and C
        wanted = {f"{m}|{C}" for m, cs in grids.items() for C in cs}
        arrays: dict[str, np.ndarray] = {}
        for fold in folds:
            saved = np.load(partial / f"{fold.tag}.npz")
            for key in wanted & set(saved.files):
                arrays.setdefault(key, np.full((repeats, data.n), np.nan))[fold.repeat, fold.test] = saved[key]
        for key, arr in arrays.items():
            model, C = key.split("|")
            target = root / model / atlas
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f"C{float(C):g}.npy", arr)
        log.info("%s %s finished (%d splits computed now)", scheme, atlas, computed)
    write_json(root / "folds.json", [{"repeat": f.repeat, "index": f.index, "test": f.test.tolist()} for f in folds])
    np.save(root / "labels.npy", data.labels)
    np.save(root / "sites.npy", data.sites.astype(str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="brainssl fmri study", description=__doc__.split("\n\n")[0])
    p.add_argument("--atlases", nargs="+", default=list(ATLASES))
    p.add_argument("--scheme", choices=["kfold", "loso"], default="kfold")
    p.add_argument("--models", nargs="+", default=["tangent", "log", "corr", "vicinal"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--baseline-c", type=float, nargs="+", default=list(BASELINE_GRID),
                   help="values of C for the tangent, log-Euclidean and correlation baselines")
    p.add_argument("--vicinal-c", type=float, nargs="+", default=list(VICINAL_GRID),
                   help="values of C for the noise-aware model")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("runs/fmri_study"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(args.out / f"settings_{args.scheme}.json", settings)
    # One atlas in memory at a time: the 392-region atlas alone needs several GB.
    for atlas in args.atlases:
        data = ConnectomeData.load(args.raw, args.processed, (atlas,), windows="vicinal" in args.models)
        folds = (kfold_splits(data.labels, data.sites, args.folds, args.repeats, args.seed) if args.scheme == "kfold"
                 else site_splits(data.sites))
        grids = {m: tuple(args.vicinal_c if m == "vicinal" else args.baseline_c) for m in args.models}
        run_study(data, (atlas,), folds, args.out, args.scheme, grids)
        del data
    log.info("study %s finished: %s", args.scheme, json.dumps(args.atlases))
