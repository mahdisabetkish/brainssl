"""Read finished runs back from disk and compute every comparison in the report.

Nothing here trains or predicts: the numbers in the README, the results page
and the dashboard are all recomputed from the files that the runs left behind
(per-case test scores for segmentation, out-of-fold probabilities for the
connectome classifiers), so they can be regenerated at any time and checked
against the artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stats import (
    auc_score,
    bootstrap_ci,
    corrected_resampled_ttest,
    paired_bootstrap,
    paired_case_test,
)

REGIONS = ("WT", "TC", "ET")


# --------------------------------------------------------------------------- segmentation

def load_smri_runs(root: Path) -> pd.DataFrame:
    """One row per evaluated fine-tuning run, from eval_test.json (and its config)."""
    rows = []
    for path in sorted(root.glob("**/eval_test.json")):
        ev = json.loads(path.read_text())
        cfg = ev.get("config") or {}
        run_dir = path.parent
        if not cfg and (run_dir / "config.json").exists():
            full = json.loads((run_dir / "config.json").read_text())
            cfg = {k: full.get(k) for k in ("bilateral", "budget", "init", "iterations", "seed")}
        tag = ev.get("run", run_dir.name)
        rows.append({"tag": tag, "dir": str(run_dir), "budget": cfg.get("budget"),
                     "bilateral": bool(cfg.get("bilateral")), "pretrained": bool(cfg.get("init")),
                     "init": cfg.get("init") or "", "seed": cfg.get("seed", 0), **ev["mean"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["budget_pct"] = df["budget"].str.replace("pct", "").astype(int)
    return df


def per_case(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "eval_test.csv").set_index("case_id")


def smri_comparisons(runs: pd.DataFrame, ours: str = "ours", baseline: str = "unet_scratch") -> list[dict]:
    """Paired per-case tests of ours against the baseline at every label budget."""
    out = []
    for budget, group in runs.groupby("budget"):
        a = group[group.tag.str.contains(ours)]
        b = group[group.tag.str.contains(baseline)]
        if a.empty or b.empty:
            continue
        ca, cb = per_case(a.iloc[0]["dir"]), per_case(b.iloc[0]["dir"])
        common = ca.index.intersection(cb.index)
        row = {"budget": budget, "cases": int(len(common)), "ours": a.iloc[0]["tag"], "baseline": b.iloc[0]["tag"]}
        for metric in ["dice_mean"] + [f"dice_{r}" for r in REGIONS]:
            test = paired_case_test(ca.loc[common, metric].to_numpy(), cb.loc[common, metric].to_numpy())
            row[metric] = test
        for metric in [f"hd95_{r}" for r in REGIONS]:
            # Lower is better for distances, so test baseline minus ours.
            test = paired_case_test(cb.loc[common, metric].to_numpy(), ca.loc[common, metric].to_numpy())
            row[metric] = test
        out.append(row)
    return out


# --------------------------------------------------------------------------- connectomes

def load_fmri_runs(root: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(root.glob("*/summary.json")):
        run_dir = path.parent
        summary = json.loads(path.read_text())
        oof = dict(np.load(run_dir / "oof.npz"))
        per_fold = json.loads((run_dir / "per_fold.json").read_text())
        per_site = None
        if (run_dir / "per_site.json").exists():
            per_site = json.loads((run_dir / "per_site.json").read_text())
        runs[run_dir.name] = {"summary": summary, "oof": oof, "per_fold": per_fold, "per_site": per_site,
                              "scheme": summary.get("scheme", "kfold"), "method": summary.get("method")}
    return runs


def fmri_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, run in runs.items():
        s = run["summary"]
        labels = run["oof"]["labels"]
        prob = run["oof"]["prob"]
        auc_ci = bootstrap_ci(auc_score, labels, np.nanmean(prob, 0))
        rows.append({"run": name, "method": run["method"], "scheme": run["scheme"],
                     **{f"{k}_mean": v for k, v in s["mean"].items()},
                     **{f"{k}_std": v for k, v in s["std"].items()},
                     "auc_ci_low": auc_ci[1], "auc_ci_high": auc_ci[2]})
    return pd.DataFrame(rows).sort_values(["scheme", "auc_mean"], ascending=[True, False])


def fmri_comparison(ours: dict, other: dict, n_subjects: int) -> dict:
    """Fold-matched corrected t-tests (k-fold) and a paired bootstrap on pooled predictions."""
    a = {(r["repeat"], r["fold"]): r for r in ours["per_fold"]}
    b = {(r["repeat"], r["fold"]): r for r in other["per_fold"]}
    keys = sorted(set(a) & set(b))
    out: dict = {"folds": len(keys)}
    if not keys:
        return out
    n_test = float(np.mean([a[k]["n_test"] for k in keys]))
    for metric in ("auc", "accuracy", "balanced_accuracy"):
        if metric in a[keys[0]] and metric in b[keys[0]]:
            diffs = np.array([a[k][metric] - b[k][metric] for k in keys])
            out[metric] = corrected_resampled_ttest(diffs, n_train=int(n_subjects - n_test), n_test=int(n_test))
            out[metric]["wins"] = int((diffs > 0).sum())
            out[metric]["losses"] = int((diffs < 0).sum())
    labels = ours["oof"]["labels"]
    pa, pb = np.nanmean(ours["oof"]["prob"], 0), np.nanmean(other["oof"]["prob"], 0)
    out["pooled_auc"] = paired_bootstrap(auc_score, labels, pa, pb)
    out["pooled_accuracy"] = paired_bootstrap(lambda y, p: float(((p >= 0.5) == y).mean()), labels, pa, pb)
    return out


def calibration_curve(labels: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & ((prob < hi) | (hi == 1.0))
        if sel.sum() >= 5:
            out.append({"bin": [float(lo), float(hi)], "n": int(sel.sum()), "mean_prob": float(prob[sel].mean()),
                        "frac_positive": float(labels[sel].mean())})
    return out
