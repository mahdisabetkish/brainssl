"""Turn the out-of-fold arrays of ``brainssl fmri study`` into reported numbers.

Choices that decide the comparison, fixed here and in assets/fmri_choices.json:

* Each baseline is reported at the regularisation strength that does best on
  the reported splits themselves (per atlas, by mean AUC over repeats). This is
  an optimistic estimate for the baselines.
* The noise-aware model is reported at the strength chosen beforehand from
  development runs on different random splits (assets/fmri_choices.json).
* Atlas ensembles average the per-atlas probabilities.
* Paired tests use the 50 shared splits (corrected resampled t-test) and a
  bootstrap over subjects on the out-of-fold probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..stats import auc_score, classification_metrics, corrected_resampled_ttest, paired_bootstrap

MODELS = {"vicinal": "Noise-aware (ours)", "tangent": "Tangent space + LR", "log": "Log-Euclidean + LR",
          "corr": "Correlation + LR"}


def load_study(root: Path, scheme: str) -> dict:
    base = root / scheme
    labels = np.load(base / "labels.npy")
    sites = np.load(base / "sites.npy")
    folds = json.loads((base / "folds.json").read_text())
    probs: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name in MODELS):
        for atlas_dir in sorted(model_dir.iterdir()):
            for f in atlas_dir.glob("C*.npy"):
                probs.setdefault(model_dir.name, {}).setdefault(atlas_dir.name, {})[float(f.stem[1:])] = np.load(f)
    return {"labels": labels, "sites": sites, "folds": folds, "probs": probs}


def per_repeat(labels: np.ndarray, prob: np.ndarray, metric: str) -> np.ndarray:
    return np.array([classification_metrics(labels, p)[metric] for p in prob])


def per_fold(labels: np.ndarray, prob: np.ndarray, folds: list[dict], metric: str) -> np.ndarray:
    out = []
    for f in folds:
        te = np.array(f["test"])
        y, p = labels[te], prob[f["repeat"], te]
        out.append(classification_metrics(y, p)[metric] if len(np.unique(y)) == 2 else np.nan)
    return np.array(out)


def select(study: dict, model: str, chosen: dict[str, float] | None) -> dict[str, tuple[float, np.ndarray]]:
    """Per atlas: (C, probabilities). Baselines: best mean AUC on these splits; ours: the pre-registered C."""
    labels = study["labels"]
    out = {}
    for atlas, by_c in study["probs"][model].items():
        if chosen is not None:
            C = chosen[atlas]
        else:
            C = max(by_c, key=lambda c: per_repeat(labels, by_c[c], "auc").mean())
        out[atlas] = (C, by_c[C])
    return out


def ensemble(selection: dict[str, tuple[float, np.ndarray]], atlases: list[str] | None = None) -> np.ndarray:
    keys = atlases or sorted(selection)
    return np.mean([selection[a][1] for a in keys], axis=0)


def summarise(labels: np.ndarray, prob: np.ndarray) -> dict:
    out = {}
    for metric in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ece", "brier"):
        values = per_repeat(labels, prob, metric)
        out[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


def compare(labels: np.ndarray, folds: list[dict], ours: np.ndarray, other: np.ndarray, n: int) -> dict:
    result = {}
    n_test = float(np.mean([len(f["test"]) for f in folds]))
    for metric in ("auc", "accuracy", "balanced_accuracy"):
        a, b = per_fold(labels, ours, folds, metric), per_fold(labels, other, folds, metric)
        ok = ~(np.isnan(a) | np.isnan(b))
        d = a[ok] - b[ok]
        if len(d) > 2:
            test = corrected_resampled_ttest(d, n_train=int(n - n_test), n_test=int(n_test))
        else:                       # too few splits with both classes in the test fold
            test = {"mean_diff": float(np.mean(d)) if len(d) else float("nan"), "t": float("nan"),
                    "p_two_sided": float("nan"), "p_one_sided": float("nan"), "folds": int(len(d))}
        ids = [f"repeat {f['repeat'] + 1}, fold {f['index'] + 1}" for f, keep in zip(folds, ok) if keep]
        result[metric] = {**test, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                          "fold_diffs": d.tolist(), "fold_ids": ids}
    pooled_a, pooled_b = np.nanmean(ours, 0), np.nanmean(other, 0)
    result["pooled_auc"] = paired_bootstrap(auc_score, labels, pooled_a, pooled_b)
    return result


def build(root: Path, choices_path: Path) -> dict:
    choices = json.loads(choices_path.read_text())
    report: dict = {"choices": choices}
    for scheme in ("kfold", "loso"):
        if not (root / scheme / "labels.npy").exists():
            continue
        study = load_study(root, scheme)
        labels, folds = study["labels"], study["folds"]
        n = len(labels)
        selections = {m: select(study, m, choices["vicinal_C"] if m == "vicinal" else None)
                      for m in study["probs"]}
        rows = []
        for model, sel in selections.items():
            for atlas, (C, prob) in sel.items():
                rows.append({"model": model, "name": MODELS[model], "atlas": atlas, "C": C, **summarise(labels, prob)})
            rows.append({"model": model, "name": MODELS[model], "atlas": "ensemble", "C": None,
                         **summarise(labels, ensemble(sel))})
        ours_ens = ensemble(selections["vicinal"])
        comparisons = {}
        for model, sel in selections.items():
            if model == "vicinal":
                continue
            comparisons[f"{model}|ensemble"] = compare(labels, folds, ours_ens, ensemble(sel), n)
            for atlas in sel:
                ours_atlas = selections["vicinal"][atlas][1]
                comparisons[f"{model}|{atlas}"] = compare(labels, folds, ours_atlas, sel[atlas][1], n)
        per_site = None
        if scheme == "loso":
            per_site = {}
            for model, sel in selections.items():
                p = ensemble(sel)[0]
                for site in np.unique(study["sites"]):
                    m = study["sites"] == site
                    per_site.setdefault(site, {"n": int(m.sum())})[model] = float(((p[m] >= 0.5) == labels[m]).mean())
        report[scheme] = {"rows": rows, "comparisons": comparisons, "per_site": per_site,
                          "ensemble_probs": {m: ensemble(s).tolist() for m, s in selections.items()},
                          "labels": labels.tolist(), "folds": folds}
    return report
