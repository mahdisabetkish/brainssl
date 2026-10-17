"""Trust audit of the connectome classifiers.

Reads the out-of-fold probabilities of ``brainssl fmri study`` (so no model is
refitted for the first four parts) and asks, for the noise-aware model and the
three linear baselines:

1. calibration, in 10-fold cross-validation and under leave-one-site-out;
2. selective prediction: how fast the error falls when the model may decline
   its least confident subjects;
3. subgroups: AUC inside every site, sex, age tercile and motion tercile;
4. shortcuts: how much of the AUC a classifier gets from the site, the head
   motion, or the age and sex alone, and how much of the model's AUC survives
   once between-site differences are removed;
5. interpretation of the shipped weights (``interpret.py`` holds the part that
   refits models).

The analysis plan is in docs/trust.md and was committed before this code was
run on the final results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..report.fmri_study import MODELS, ensemble, load_study, select
from ..stats import auc_score, classification_metrics
from ..utils import write_json
from .measures import (
    calibration_slope_intercept,
    crossfit_platt,
    rank_correlation,
    reliability_bins,
    risk_coverage,
    subgroup_auc,
    within_group_auc,
)

log = logging.getLogger(__name__)


def _ensembles(root: Path, scheme: str, choices: dict) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list]:
    study = load_study(root, scheme)
    probs = {m: ensemble(select(study, m, choices["vicinal_C"] if m == "vicinal" else None))
             for m in study["probs"]}
    return probs, study["labels"], study["sites"], study["folds"]


def calibration_block(y: np.ndarray, prob: np.ndarray) -> dict:
    """Calibration of one model; ``prob`` is (repeats, subjects)."""
    per = [classification_metrics(y, p) for p in prob]
    slopes = [calibration_slope_intercept(y, p) for p in prob]
    stacked_y, stacked_p = np.tile(y, prob.shape[0]), prob.reshape(-1)
    return {"ece": _ms([m["ece"] for m in per]), "brier": _ms([m["brier"] for m in per]),
            "slope": _ms([s["slope"] for s in slopes]), "intercept": _ms([s["intercept"] for s in slopes]),
            "bins": reliability_bins(stacked_y, stacked_p, 10),
            "mean_confidence": float(np.mean(np.where(stacked_p >= 0.5, stacked_p, 1 - stacked_p))),
            "accuracy": float(np.mean((stacked_p >= 0.5) == stacked_y))}


def _ms(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _fold_groups(folds: list[dict], n: int, repeats: int) -> np.ndarray:
    groups = np.zeros((repeats, n), dtype=int)
    for f in folds:
        groups[f["repeat"], np.array(f["test"])] = f["index"]
    return groups


def recalibration_block(y: np.ndarray, prob: np.ndarray, groups: np.ndarray) -> dict:
    """Raw against cross-fitted recalibrated probabilities, and the map to ship.

    ``groups`` has the shape of ``prob`` and gives the fold (or site) of every
    out-of-fold prediction; the map applied to a subject is fitted on the
    predictions of the other groups only.
    """
    fixed = np.stack([crossfit_platt(y, p, g) for p, g in zip(prob, groups)])

    def row(q: np.ndarray) -> dict:
        per = [classification_metrics(y, p) for p in q]
        return {k: _ms([m[k] for m in per])
                for k in ("ece", "brier", "accuracy", "balanced_accuracy", "auc")} | {
                    "slope": _ms([calibration_slope_intercept(y, p)["slope"] for p in q])}

    ship = calibration_slope_intercept(np.tile(y, prob.shape[0]), prob.reshape(-1))
    return {"raw": row(prob), "recalibrated": row(fixed), "ship": ship,
            "bins_recalibrated": reliability_bins(np.tile(y, prob.shape[0]), fixed.reshape(-1), 10)}


def selective_block(y: np.ndarray, prob: np.ndarray) -> dict:
    """Risk-coverage with error as the loss and |p - 1/2| as the confidence, averaged over repeats."""
    curves = [risk_coverage((((p >= 0.5).astype(int)) != y).astype(float), np.abs(p - 0.5)) for p in prob]
    mean = lambda k: np.mean([c[k] for c in curves], axis=0).tolist()          # noqa: E731
    out = {"coverage": curves[0]["coverage"], "risk": mean("risk"), "oracle_risk": mean("oracle_risk"),
           "aurc": _ms([c["aurc"] for c in curves]), "excess": _ms([c["excess"] for c in curves])}
    cov = np.array(out["coverage"])
    risk = np.array(out["risk"])
    for target in (0.5, 0.8):
        out[f"accuracy_at_{int(target * 100)}"] = float(1 - risk[np.searchsorted(cov, target - 1e-9)])
    out["accuracy_full"] = float(1 - risk[-1])
    return out


def subgroup_block(y: np.ndarray, p: np.ndarray, pheno: pd.DataFrame) -> dict:
    """AUC by site, sex, age tercile and motion tercile; ``p`` is one probability per subject."""
    # ranks first, so that ties cannot make two terciles collapse
    age = pd.qcut(pheno["age"].rank(method="first"), 3,
                  labels=["youngest third", "middle third", "oldest third"]).astype(str)
    fd = pd.qcut(pheno["func_mean_fd"].rank(method="first"), 3,
                 labels=["least motion", "middle third", "most motion"]).astype(str)
    sex = np.where(pheno["sex"] == 1, "female", "male")
    out = {"site": subgroup_auc(y, p, pheno["site"].to_numpy()),
           "sex": subgroup_auc(y, p, sex), "age": subgroup_auc(y, p, age.to_numpy()),
           "motion": subgroup_auc(y, p, fd.to_numpy())}
    sites = [r for r in out["site"] if r["auc"] is not None]
    out["site_summary"] = {"sites_scored": len(sites), "sites_total": len(out["site"]),
                           "worst_auc": min((r["auc"] for r in sites), default=None),
                           "best_auc": max((r["auc"] for r in sites), default=None),
                           "median_auc": float(np.median([r["auc"] for r in sites])) if sites else None,
                           "sites_ci_above_half": int(sum(r["ci_low"] > 0.5 for r in sites))}
    return out


def _nuisance_design(pheno: pd.DataFrame, kind: str) -> np.ndarray:
    site = pd.get_dummies(pheno["site"]).to_numpy(float)
    motion = np.log(pheno["func_mean_fd"].to_numpy(float))[:, None]
    demo = np.stack([pheno["age"].to_numpy(float), pheno["sex"].to_numpy(float)], axis=1)
    parts = {"site": [site], "motion": [motion], "age and sex": [demo],
             "site, motion, age and sex": [site, motion, demo]}[kind]
    x = np.concatenate(parts, axis=1)
    keep = x.std(0) > 0
    x = x[:, keep]
    mu, sd = x.mean(0), x.std(0)
    return np.where(sd > 0, (x - mu) / np.where(sd > 0, sd, 1), x)


def shortcut_block(y: np.ndarray, prob: np.ndarray, pheno: pd.DataFrame, folds: list[dict]) -> dict:
    """Nuisance-only classifiers on the very same folds, and the model's score with nuisance removed."""
    n = len(y)
    out: dict = {"nuisance_only": []}
    repeats = 1 + max(f["repeat"] for f in folds)
    for kind in ("site", "motion", "age and sex", "site, motion, age and sex"):
        x = _nuisance_design(pheno, kind)
        oof = np.full((repeats, n), np.nan)
        for f in folds:
            te = np.array(f["test"])
            tr = np.setdiff1d(np.arange(n), te)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(x[tr], y[tr])
            oof[f["repeat"], te] = clf.predict_proba(x[te])[:, 1]
        aucs = [auc_score(y, p) for p in oof]
        out["nuisance_only"].append({"inputs": kind, "auc": _ms(aucs)})
    sites = pheno["site"].to_numpy()
    fd = pheno["func_mean_fd"].to_numpy(float)
    tercile = pd.qcut(pd.Series(fd).rank(method="first"), 3, labels=False).to_numpy()
    out["model"] = {
        "auc": _ms([auc_score(y, p) for p in prob]),
        "auc_within_site": _ms([within_group_auc(y, p, sites) for p in prob]),
        "auc_within_motion_tercile": _ms([within_group_auc(y, p, tercile) for p in prob]),
        "auc_within_site_and_motion": _ms([within_group_auc(y, p, np.char.add(sites.astype(str), tercile.astype(str)),
                                                            min_per_class=3) for p in prob]),
        "score_vs_motion_controls": rank_correlation(prob.mean(0)[y == 0], fd[y == 0], n_boot=500),
        "score_vs_motion_autism": rank_correlation(prob.mean(0)[y == 1], fd[y == 1], n_boot=500),
        "score_vs_age_controls": rank_correlation(prob.mean(0)[y == 0], pheno["age"].to_numpy(float)[y == 0],
                                                  n_boot=500),
    }
    return out


def audit(study_root: Path, choices_path: Path, raw: Path, out: Path | None = None) -> dict:
    """Parts 1 to 4 of the fMRI trust audit."""
    from ..fmri.abide import load_phenotypes
    choices = json.loads(choices_path.read_text())
    pheno = load_phenotypes(raw / "abide")
    report: dict = {"models": {m: MODELS[m] for m in MODELS}}
    for scheme in ("kfold", "loso"):
        if not (study_root / scheme / "labels.npy").exists():
            continue
        probs, y, sites, folds = _ensembles(study_root, scheme, choices)
        if len(y) != len(pheno) or not np.array_equal(y, pheno["label"].to_numpy()):
            raise ValueError("study labels do not match the phenotype table")
        block = {"calibration": {m: calibration_block(y, p) for m, p in probs.items()},
                 "selective": {m: selective_block(y, p) for m, p in probs.items()}}
        if scheme == "kfold":
            groups = _fold_groups(folds, len(y), next(iter(probs.values())).shape[0])
        else:                       # under leave-one-site-out the map must be fitted on the other sites
            groups = np.tile(sites.astype(str), (next(iter(probs.values())).shape[0], 1))
        block["recalibration"] = {m: recalibration_block(y, p, groups) for m, p in probs.items()}
        if scheme == "kfold":
            mean_p = {m: p.mean(0) for m, p in probs.items()}
            block["subgroups"] = {m: subgroup_block(y, p, pheno) for m, p in mean_p.items() if m in ("vicinal", "log")}
            block["shortcuts"] = shortcut_block(y, probs["vicinal"], pheno, folds)
            block["shortcuts"]["baseline_within_site"] = {
                m: _ms([within_group_auc(y, p, sites.astype(str)) for p in probs[m]]) for m in probs}
        report[scheme] = block
        log.info("fMRI trust, %s done", scheme)
    if out is not None:
        write_json(out, report)
    return report
