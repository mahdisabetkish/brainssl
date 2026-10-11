"""Metrics, confidence intervals and paired tests used in the reports.

Two kinds of comparison appear in this repository and they need different
tools. Segmentation models are compared case by case on one fixed test set, so
paired differences per case support a Wilcoxon signed-rank test and a bootstrap
interval for the mean difference. Connectome classifiers are compared across
repeated cross-validation, where the training sets of different folds overlap
and the usual t-test is badly anti-conservative; the corrected resampled t-test
of Nadeau and Bengio (Machine Learning 52, 2003) inflates the variance by the
test/train size ratio to account for that.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE of the predicted probability of the positive class, equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties sharing their mean rank (scipy's rankdata, 'average')."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    edges = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])
    ranks = np.empty(len(x))
    ranks[order] = np.repeat((edges[:-1] + edges[1:] + 1) / 2, np.diff(edges))
    return ranks


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve from the Mann-Whitney statistic (ties count one half).

    Same value as sklearn's roc_auc_score at a fraction of the cost, which
    matters inside bootstrap loops.
    """
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    ranks = _average_ranks(np.asarray(p, dtype=float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Threshold metrics, AUC, Brier score and ECE of predicted probabilities for class 1."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos, neg = tp + fn, tn + fp
    recalls = [r for r, m in ((tp / max(1, pos), pos), (tn / max(1, neg), neg)) if m > 0]
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": float(np.mean(recalls)),
        "auc": auc_score(y, p),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "sensitivity": tp / max(1, pos),
        "specificity": tn / max(1, neg),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_ci(fn: Callable[..., float], *arrays: np.ndarray, n_boot: int = 2000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``fn(*arrays)`` over resampled rows."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = fn(*arrays)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            draws.append(fn(*(a[idx] for a in arrays)))
        except ValueError:          # e.g. a resample with one class only
            continue
    lo, hi = np.percentile(draws, [50 * (1 - level), 50 * (1 + level)])
    return float(point), float(lo), float(hi)


def paired_bootstrap(fn: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, p_a: np.ndarray,
                     p_b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    """Difference fn(y, p_a) - fn(y, p_b) with a percentile interval and a one-sided p-value
    for the hypothesis that model A is no better than model B."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diff = fn(y, p_a) - fn(y, p_b)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            draws.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
        except ValueError:
            continue
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(diff), "ci_low": float(lo), "ci_high": float(hi),
            "p_one_sided": float((np.sum(draws <= 0) + 1) / (len(draws) + 1))}


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau-Bengio test on per-fold score differences from repeated k-fold CV."""
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / k + n_test / n_train) * var)
    t = mean / se if se > 0 else np.inf
    p_two = 2 * stats.t.sf(abs(t), df=k - 1)
    p_one = stats.t.sf(t, df=k - 1)
    return {"mean_diff": float(mean), "t": float(t), "p_two_sided": float(p_two),
            "p_one_sided": float(p_one), "folds": k}


def paired_case_test(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 5000,
                     seed: int = 0) -> dict[str, float]:
    """Per-case paired comparison (e.g. Dice of two models on the same test cases)."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    nonzero = d[d != 0]
    w = stats.wilcoxon(nonzero, alternative="greater") if len(nonzero) else None
    return {
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "wilcoxon_p_one_sided": float(w.pvalue) if w is not None else 1.0,
    }
