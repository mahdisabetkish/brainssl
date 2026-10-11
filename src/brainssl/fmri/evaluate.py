"""Cross-validation protocol shared by every connectome classifier.

All methods are scored on exactly the same splits:

* repeated stratified k-fold (default 5 x 10), stratified on site and diagnosis
  together so every fold keeps each site's case/control balance;
* leave-one-site-out, the harder test of whether a model transfers to a
  scanner and population it has never seen.

A run stores out-of-fold probabilities for every repeat, so any metric, any
paired test and the dashboard can be computed afterwards without retraining.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from ..stats import classification_metrics
from ..utils import write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    repeat: int
    index: int
    train: np.ndarray
    test: np.ndarray

    @property
    def tag(self) -> str:
        return f"r{self.repeat}f{self.index}"


def kfold_splits(labels: np.ndarray, sites: np.ndarray, n_splits: int = 10, repeats: int = 5,
                 seed: int = 0) -> list[Fold]:
    strata = np.array([f"{s}|{l}" for s, l in zip(sites, labels)])
    folds = []
    with warnings.catch_warnings():
        # The smallest site-by-diagnosis stratum has five subjects; scikit-learn
        # warns that it cannot appear in every one of ten folds, which is expected.
        warnings.filterwarnings("ignore", message="The least populated class")
        for r in range(repeats):
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
            for k, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), strata)):
                folds.append(Fold(r, k, tr, te))
    return folds


def site_splits(sites: np.ndarray) -> list[Fold]:
    folds = []
    for k, (tr, te) in enumerate(LeaveOneGroupOut().split(np.zeros(len(sites)), groups=sites)):
        folds.append(Fold(0, k, tr, te))
    return folds


class Method(Protocol):
    name: str

    def fit_predict(self, fold: Fold) -> np.ndarray | dict[str, np.ndarray]:
        """Probabilities for ``fold.test``; a dict may add e.g. an uncertainty ``std``."""


def run_cv(method: Method, folds: list[Fold], labels: np.ndarray, out_dir: Path,
           scheme: str = "kfold") -> dict:
    """Fit and score ``method`` on every fold, resuming from saved folds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = 1 + max(f.repeat for f in folds)
    n = len(labels)
    store = out_dir / "folds"
    store.mkdir(exist_ok=True)
    probs = np.full((repeats, n), np.nan)
    extra: dict[str, np.ndarray] = {}
    rows = []
    t0 = time.time()
    for fold in folds:
        path = store / f"{fold.tag}.npz"
        if path.exists():
            saved = dict(np.load(path))
        else:
            out = method.fit_predict(fold)
            saved = dict(out) if isinstance(out, dict) else {"prob": out}
            saved = {k: np.asarray(v, dtype=np.float64) for k, v in saved.items()}
            np.savez(path, **saved)
        probs[fold.repeat, fold.test] = saved["prob"]
        for k, v in saved.items():
            if k != "prob":
                extra.setdefault(k, np.full((repeats, n), np.nan))[fold.repeat, fold.test] = v
        m = classification_metrics(labels[fold.test], saved["prob"]) if len(np.unique(labels[fold.test])) == 2 \
            else {"accuracy": float(((saved["prob"] >= 0.5) == labels[fold.test]).mean())}
        rows.append({"repeat": fold.repeat, "fold": fold.index, "n_test": int(len(fold.test)), **m})
    per_repeat = [classification_metrics(labels, probs[r]) for r in range(repeats)]
    summary = {
        "method": method.name, "scheme": scheme, "folds": len(folds), "repeats": repeats,
        "mean": {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]},
        "std": {k: float(np.std([m[k] for m in per_repeat])) for k in per_repeat[0]},
        "seconds": time.time() - t0,
    }
    np.savez(out_dir / "oof.npz", prob=probs, labels=labels, **extra)
    write_json(out_dir / "per_fold.json", rows)
    write_json(out_dir / "summary.json", summary)
    log.info("%s [%s]: acc %.3f +- %.3f, auc %.3f", method.name, scheme, summary["mean"]["accuracy"],
             summary["std"]["accuracy"], summary["mean"]["auc"])
    return summary


def load_run(out_dir: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    summary = json.loads((out_dir / "summary.json").read_text())
    oof = dict(np.load(out_dir / "oof.npz"))
    per_fold = json.loads((out_dir / "per_fold.json").read_text())
    return summary, oof, per_fold
