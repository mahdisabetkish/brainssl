import numpy as np
import pytest

from brainssl.trust.measures import (
    agreement,
    block_sums,
    calibration_slope_intercept,
    crossfit_platt,
    platt_apply,
    rank_correlation,
    reliability_bins,
    risk_coverage,
    subgroup_auc,
    top_k_overlap,
    within_group_auc,
)


def test_reliability_bins_of_a_calibrated_predictor(rng):
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    bins = reliability_bins(y, p, bins=10)
    assert sum(b["n"] for b in bins) == p.size
    assert all(abs(b["mean_pred"] - b["observed"]) < 0.05 for b in bins)


def test_calibration_slope_detects_overconfidence(rng):
    p_true = rng.uniform(0.1, 0.9, 20000)
    y = (rng.uniform(0, 1, p_true.size) < p_true).astype(int)
    ok = calibration_slope_intercept(y, p_true)
    assert ok["slope"] == pytest.approx(1.0, abs=0.08) and ok["intercept"] == pytest.approx(0.0, abs=0.05)
    logit = np.log(p_true / (1 - p_true))
    sharp = 1 / (1 + np.exp(-3 * logit))              # same ranking, predictions far too extreme
    assert calibration_slope_intercept(y, sharp)["slope"] == pytest.approx(1 / 3, abs=0.06)


def test_risk_coverage_bounds_and_ordering(rng):
    loss = rng.uniform(0, 1, 400)
    perfect = risk_coverage(loss, -loss)              # confidence is exactly the negative loss
    blind = risk_coverage(loss, rng.uniform(0, 1, 400))
    worst = risk_coverage(loss, loss)
    assert perfect["excess"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["aurc"] == pytest.approx(perfect["oracle_aurc"])
    assert 0.6 < blind["excess"] < 1.4
    assert worst["excess"] > 1.5
    assert perfect["risk"][-1] == pytest.approx(loss.mean())      # full coverage is the overall mean
    assert np.all(np.diff(perfect["coverage"]) > 0)


def test_within_group_auc_ignores_a_site_shortcut(rng):
    n = 600
    site = np.repeat(np.arange(4), n // 4)
    y = (rng.uniform(0, 1, n) < 0.2 + 0.2 * site).astype(int)       # prevalence differs by site
    shortcut = site + rng.normal(0, 0.01, n)                         # knows only the site
    from brainssl.stats import auc_score
    assert auc_score(y, shortcut) > 0.6
    assert within_group_auc(y, shortcut, site) == pytest.approx(0.5, abs=0.08)
    real = y + rng.normal(0, 0.5, n)
    assert within_group_auc(y, real, site) > 0.75


def test_subgroup_auc_skips_small_groups(rng):
    y = np.r_[np.zeros(30), np.ones(30), np.zeros(3), np.ones(2)].astype(int)
    p = np.r_[rng.uniform(0, 0.6, 30), rng.uniform(0.4, 1, 30), 0.5, 0.5, 0.5, 0.5, 0.5]
    g = np.r_[np.zeros(60), np.ones(5)]
    rows = subgroup_auc(y, p, g, min_per_class=8, n_boot=100)
    assert rows[0]["auc"] > 0.8 and rows[0]["ci_low"] <= rows[0]["auc"] <= rows[0]["ci_high"]
    assert rows[1]["auc"] is None and rows[1]["n"] == 5


def test_agreement_of_a_biased_measurement(rng):
    ref = rng.uniform(10, 100, 500)
    pred = 1.1 * ref + 2 + rng.normal(0, 1, 500)
    a = agreement(pred, ref)
    assert a["pearson"] > 0.99
    assert a["bias"] > 0 and a["loa_low"] < a["bias"] < a["loa_high"]
    assert a["relative_error"] == pytest.approx(0.1, abs=0.04)


def test_rank_correlation_has_sensible_interval(rng):
    a = rng.normal(size=300)
    r = rank_correlation(a, a + rng.normal(0, 0.5, 300), n_boot=200)
    assert 0.6 < r["rho"] < 0.95 and r["ci_low"] < r["rho"] < r["ci_high"]


def test_block_sums_count_every_unordered_pair_once():
    m = np.arange(16, dtype=float).reshape(4, 4)
    m = m + m.T                                        # symmetric
    groups = np.array(["a", "a", "b", "b"])
    out = block_sums(m, groups)
    assert out["names"] == ["a", "b"]
    # 4 ROIs have 10 entries on or above the diagonal, and every one is counted once
    assert out["count"][0, 0] + out["count"][1, 1] + out["count"][0, 1] == 10
    assert np.triu(out["sum"]).sum() == pytest.approx(np.triu(m).sum())
    assert out["sum"][0, 1] == out["sum"][1, 0]
    assert out["sum"][0, 1] == pytest.approx(m[:2, 2:].sum())


def test_top_k_overlap():
    a = np.array([5, -4, 3, 0.1, 0.2])
    assert top_k_overlap(a, a, 3) == 1.0
    assert top_k_overlap(a, np.array([0, 0, 0, 9, 8]), 2) == 0.0


def test_crossfit_platt_repairs_timid_predictions_and_keeps_the_ranking(rng):
    from brainssl.stats import auc_score, expected_calibration_error
    p_true = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(0, 1, p_true.size) < p_true).astype(int)
    timid = platt_apply(p_true, 0.0, 0.4)                       # pulled towards one half
    folds = np.arange(p_true.size) % 10
    fixed = crossfit_platt(y, timid, folds)
    assert expected_calibration_error(y, fixed) < 0.5 * expected_calibration_error(y, timid)
    assert calibration_slope_intercept(y, fixed)["slope"] == pytest.approx(1.0, abs=0.12)
    # one map per fold: the ranking is kept inside each fold, so the pooled AUC moves only a little
    assert auc_score(y, fixed) == pytest.approx(auc_score(y, timid), abs=0.01)
