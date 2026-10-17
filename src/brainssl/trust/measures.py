"""Numerical measures behind the trust report.

Everything here is a plain function of arrays, so it can be tested on small
examples whose answers are known. The two audits (``fmri.py``, ``smri.py``)
only decide which arrays to feed in.

Four families of measures appear:

* calibration: do predicted probabilities mean what they say (reliability
  bins, calibration slope and intercept);
* selective prediction: if the model may decline its least confident cases, how
  fast does the error fall (risk-coverage curve and its area);
* subgroup performance: does the score hold up inside every site, sex, age and
  motion stratum, not only on average;
* agreement of scalar summaries, such as predicted against true tumour volume
  (bias and limits of agreement).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from ..stats import auc_score, bootstrap_ci


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Equal-mass bins of the predicted probability of class 1.

    Equal mass, not equal width, so that every point of the curve rests on about
    the same number of subjects. Each entry holds the mean prediction, the
    observed fraction of class 1 and the bin size.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    out = []
    for idx in np.array_split(order, bins):
        if len(idx):
            out.append({"mean_pred": float(p[idx].mean()), "observed": float(y[idx].mean()), "n": int(len(idx))})
    return out


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """Logistic regression of the outcome on the logit of the prediction.

    A calibrated model has slope 1 and intercept 0. A slope below 1 means the
    predictions are too extreme, above 1 too timid. Fitted by Newton's method
    on the two parameters, with no penalty.
    """
    y = np.asarray(y, float)
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    X = np.stack([np.ones_like(z), z], axis=1)
    beta = np.zeros(2)
    for _ in range(50):
        q = 1.0 / (1.0 + np.exp(-X @ beta))
        grad = X.T @ (q - y)
        hess = X.T @ (X * (q * (1 - q))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def platt_apply(p: np.ndarray, intercept: float, slope: float, eps: float = 1e-6) -> np.ndarray:
    """sigmoid(intercept + slope * logit(p))."""
    q = np.clip(np.asarray(p, float), eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(intercept + slope * np.log(q / (1 - q)))))


def crossfit_platt(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recalibrated probabilities where no subject helps to calibrate itself.

    The two-parameter logistic map of ``calibration_slope_intercept`` is fitted
    on the out-of-fold predictions of every subject outside a group and applied
    to the subjects inside it, for each group in turn. The groups are folds, or
    sites when calibration has to survive a change of site. Each map is increasing, so the ranking inside a
    group is unchanged and the pooled AUC moves only by the small differences
    between the maps of different groups.
    """
    y, p, groups = np.asarray(y), np.asarray(p, float), np.asarray(groups)
    out = np.empty_like(p)
    for g in set(groups.tolist()):
        m = groups == g
        fit = calibration_slope_intercept(y[~m], p[~m])
        out[m] = platt_apply(p[m], fit["intercept"], fit["slope"])
    return out


def risk_coverage(loss: np.ndarray, confidence: np.ndarray, grid: int = 20) -> dict[str, object]:
    """Mean loss of the retained cases when only the most confident are kept.

    ``loss`` is any per-case cost where lower is better (0/1 error, 1 - Dice).
    Cases are sorted by decreasing ``confidence``; the curve gives the mean loss
    over the top ``c`` fraction for c from 1/grid to 1. ``aurc`` is the area
    under that curve (lower is better). ``random_aurc`` is the loss of keeping
    cases in random order, which is the overall mean, and ``oracle_aurc`` the
    area when cases are ordered by their true loss, the best any confidence
    could do. ``excess`` is aurc minus oracle_aurc divided by random minus
    oracle: 0 is perfect ranking, 1 is no better than chance.
    """
    loss, confidence = np.asarray(loss, float), np.asarray(confidence, float)
    n = len(loss)
    order = np.argsort(-confidence, kind="mergesort")
    cum = np.cumsum(loss[order]) / np.arange(1, n + 1)
    oracle = np.cumsum(np.sort(loss)) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    marks = np.unique(np.clip((np.arange(1, grid + 1) / grid * n).round().astype(int), 1, n))
    aurc, oracle_aurc, rand = float(cum.mean()), float(oracle.mean()), float(loss.mean())
    excess = (aurc - oracle_aurc) / (rand - oracle_aurc) if rand > oracle_aurc + 1e-12 else 0.0
    return {"coverage": coverage[marks - 1].tolist(), "risk": cum[marks - 1].tolist(),
            "oracle_risk": oracle[marks - 1].tolist(), "aurc": aurc, "random_aurc": rand,
            "oracle_aurc": oracle_aurc, "excess": float(excess)}


def subgroup_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray, min_per_class: int = 8,
                 n_boot: int = 1000, seed: int = 0) -> list[dict[str, object]]:
    """AUC with a bootstrap interval inside every group that has enough of both classes."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist()), key=str):
        m = groups == g
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        row: dict[str, object] = {"group": str(g), "n": int(m.sum()), "positives": pos}
        if min(pos, neg) >= min_per_class:
            point, lo, hi = bootstrap_ci(auc_score, y[m], p[m], n_boot=n_boot, seed=seed)
            row.update({"auc": point, "ci_low": lo, "ci_high": hi})
        else:
            row.update({"auc": None, "ci_low": None, "ci_high": None})
        rows.append(row)
    return rows


def within_group_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray, min_per_class: int = 5) -> float:
    """Average of the AUC inside each group, weighted by the number of case-control pairs.

    This is the AUC with every between-group difference removed: a score that
    only separates sites (a shortcut) scores 0.5 here, while a score that
    separates diagnosis inside each site keeps its value.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    total, weight = 0.0, 0.0
    for g in set(groups.tolist()):
        m = groups == g
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        if min(pos, neg) >= min_per_class:
            total += auc_score(y[m], p[m]) * pos * neg
            weight += pos * neg
    return float(total / weight) if weight else float("nan")


def agreement(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Bland-Altman summary of a scalar measured two ways.

    ``bias`` is the mean of pred - ref, ``loa_low`` and ``loa_high`` the 95%
    limits of agreement (bias +- 1.96 sd), ``relative_error`` the median of
    |pred - ref| / ref over cases with ref > 0, and ``pearson`` / ``spearman``
    the correlations.
    """
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    d = pred - ref
    sd = float(d.std(ddof=1))
    pos = ref > 0
    varied = len(d) > 2 and pred.std() > 0 and ref.std() > 0
    return {"bias": float(d.mean()), "sd": sd, "loa_low": float(d.mean() - 1.96 * sd),
            "loa_high": float(d.mean() + 1.96 * sd),
            "relative_error": float(np.median(np.abs(d[pos]) / ref[pos])) if pos.any() else float("nan"),
            "pearson": float(sps.pearsonr(pred, ref)[0]) if varied else float("nan"),
            "spearman": float(sps.spearmanr(pred, ref)[0]) if len(d) > 2 else float("nan")}


def rank_correlation(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Spearman correlation with a bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(sps.spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        r = sps.spearmanr(a[i], b[i])[0] if a[i].std() > 0 and b[i].std() > 0 else float("nan")
        if np.isfinite(r):
            draws.append(r)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
    return {"rho": point, "ci_low": float(lo), "ci_high": float(hi)}


def block_sums(values: np.ndarray, groups: np.ndarray, labels: list[str] | None = None) -> dict[str, object]:
    """Sum of a symmetric matrix over blocks of ROIs.

    ``values`` is (R, R), ``groups`` assigns each ROI to a group. Entry (a, b) of
    the result is the sum over ROI pairs with one ROI in a and the other in b,
    counting each unordered pair once (the diagonal block counts pairs inside a
    group plus the ROIs' own diagonal entries). ``counts`` gives the number of
    entries in each block, so that a mean per entry can be formed.
    """
    groups = np.asarray(groups)
    names = labels or sorted(set(groups.tolist()), key=str)
    index = {g: i for i, g in enumerate(names)}
    k = len(names)
    code = np.array([index[g] for g in groups])
    i, j = np.triu_indices(len(groups))
    a, b = np.minimum(code[i], code[j]), np.maximum(code[i], code[j])
    sums, counts = np.zeros((k, k)), np.zeros((k, k))
    np.add.at(sums, (a, b), np.asarray(values)[i, j])
    np.add.at(counts, (a, b), 1)
    sums = sums + np.triu(sums, 1).T
    counts = counts + np.triu(counts, 1).T
    return {"names": list(names), "sum": sums, "count": counts}


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the k largest entries (by magnitude) of two vectors."""
    ia = set(np.argsort(-np.abs(a))[:k].tolist())
    ib = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ia & ib) / len(ia | ib)
