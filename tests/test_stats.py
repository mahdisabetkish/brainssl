import numpy as np

from brainssl.stats import (
    auc_score,
    bootstrap_ci,
    classification_metrics,
    corrected_resampled_ttest,
    expected_calibration_error,
    paired_case_test,
)


def test_corrected_ttest_is_more_conservative_than_naive(rng):
    d = 0.01 + 0.02 * rng.normal(size=50)
    corrected = corrected_resampled_ttest(d, n_train=780, n_test=87)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(corrected["t"]) < abs(naive_t)
    assert 0 <= corrected["p_one_sided"] <= 1


def test_paired_case_test_direction():
    a = np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.75])
    b = a - 0.05
    out = paired_case_test(a, b, n_boot=500)
    assert out["mean_diff"] > 0 and out["wins"] == 6 and out["losses"] == 0
    assert out["ci_low"] > 0
    assert out["wilcoxon_p_one_sided"] < 0.05


def test_ece_of_calibrated_predictions_is_small(rng):
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_bootstrap_interval_contains_point(rng):
    x = rng.normal(size=200)
    point, lo, hi = bootstrap_ci(np.mean, x, n_boot=500)
    assert lo <= point <= hi


def test_classification_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0 and m["balanced_accuracy"] == 1.0


def test_auc_score_matches_sklearn_with_ties(rng):
    from sklearn.metrics import roc_auc_score

    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300) + 0.3 * y, 1)      # coarse values, many ties
    assert abs(auc_score(y, p) - roc_auc_score(y, p)) < 1e-12


def test_classification_metrics_match_sklearn(rng):
    from sklearn import metrics

    y = rng.integers(0, 2, 200)
    p = np.clip(rng.random(200) * 0.8 + 0.2 * y, 0, 1)
    m = classification_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert abs(m["accuracy"] - metrics.accuracy_score(y, pred)) < 1e-12
    assert abs(m["balanced_accuracy"] - metrics.balanced_accuracy_score(y, pred)) < 1e-12
    assert abs(m["f1"] - metrics.f1_score(y, pred)) < 1e-12
    assert abs(m["auc"] - metrics.roc_auc_score(y, p)) < 1e-12
    assert abs(m["brier"] - metrics.brier_score_loss(y, p)) < 1e-12
