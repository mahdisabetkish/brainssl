"""``brainssl report``: results tables, paired tests and dashboard data.

    brainssl report --smri-runs runs/smri --study runs/fmri_study --deep-runs runs/fmri --out report

Writes ``results.md`` (the README tables are copied from it) and
``results.json`` (every number, for the dashboard and the model cards). Run
directories are read, never modified.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import read_json, write_json
from .collect import REGIONS, load_fmri_runs, load_smri_runs, smri_comparisons
from .fmri_study import MODELS, build, compare, summarise

log = logging.getLogger(__name__)

DEEP = {"mlp_cc200_kfold": "MLP (cc200)", "brainnetcnn_cc200_kfold": "BrainNetCNN (cc200)",
        "bnt_cc200_kfold": "Brain Network Transformer (cc200)"}


def _f(x, d: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _p(p: float) -> str:
    if p is None or np.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _d(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:+.3f}"


def smri_section(runs: pd.DataFrame, comparisons: list[dict]) -> list[str]:
    lines = ["## Tumour segmentation (MSD Task01, 96 held-out test scans)", ""]
    if runs.empty:
        return lines + ["No finished runs yet.", ""]
    lines += ["| Run | Labels | Dice WT | Dice TC | Dice ET | Dice mean | HD95 WT | HD95 TC | HD95 ET |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in runs.sort_values(["budget_pct", "tag"]).iterrows():
        lines.append(f"| {r.tag} | {r.budget_pct}% | " + " | ".join(_f(r[f'dice_{x}']) for x in REGIONS)
                     + f" | {_f(r.dice_mean)} | " + " | ".join(_f(r[f'hd95_{x}'], 1) for x in REGIONS) + " |")
    lines += ["", "Paired per-case comparison with the U-Net trained from scratch on the same scans "
              "(mean difference, 95% bootstrap interval, one-sided Wilcoxon signed-rank p):", "",
              "| Labels | Dice mean | Dice WT | Dice TC | Dice ET | scans better / worse |",
              "|---|---|---|---|---|---|"]
    for c in sorted(comparisons, key=lambda c: int(c["budget"].replace("pct", ""))):
        cells = []
        for m in ["dice_mean"] + [f"dice_{x}" for x in REGIONS]:
            t = c[m]
            cells.append(f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}], "
                         f"p {_p(t['wilcoxon_p_one_sided'])}")
        dm = c["dice_mean"]
        budget = c["budget"].replace("pct", "%")
        lines.append(f"| {budget} | " + " | ".join(cells) + f" | {dm['wins']} / {dm['losses']} |")
    return lines + [""]


def deep_rows(deep: dict, labels: np.ndarray, folds: list[dict], ours_prob: np.ndarray, n: int) -> tuple[list, dict]:
    rows, comps = [], {}
    for key, name in DEEP.items():
        if key not in deep:
            continue
        prob = deep[key]["oof"]["prob"]
        rows.append({"model": key, "name": name, "atlas": "cc200", "C": None, **summarise(labels, prob)})
        comps[key] = compare(labels, folds, ours_prob, prob, n)
    return rows, comps


def same_splits(run_dir: Path, settings: dict) -> bool:
    """A deep run is compared fold by fold only if it used the study's splits."""
    if not (run_dir / "signature.json").exists():
        return False
    sig = read_json(run_dir / "signature.json")
    return sig.get("scheme") == "kfold" and all(sig.get(k) == settings.get(k) for k in ("repeats", "folds", "seed"))


def fmri_section(study: dict, deep_table: list, deep_comp: dict) -> list[str]:
    lines = ["## Autism vs control from resting-state connectomes (ABIDE I, 871 subjects)", ""]
    kf = study.get("kfold")
    if not kf:
        return lines + ["No finished runs yet.", ""]
    choices = study["choices"]
    repeats = 1 + max(f["repeat"] for f in kf["folds"])
    k = len(kf["folds"]) // repeats
    lines += [f"Noise-aware model at the C fixed from development runs ({choices['rule']}); "
              f"baselines at the C that does best on these splits. Mean ± standard deviation over {repeats} "
              f"repeats of {k}-fold cross-validation.", "",
              "| Model | Atlas | C | AUC | Accuracy | Balanced acc. | Sensitivity | Specificity |",
              "|---|---|---|---|---|---|---|---|"]
    order = ["vicinal", "tangent", "log", "corr"]
    rows = sorted(kf["rows"], key=lambda r: (order.index(r["model"]), r["atlas"] != "ensemble", r["atlas"]))
    for r in rows + deep_table:
        cval = "" if r["C"] is None else f"{r['C']:g}"
        lines.append(f"| {r['name']} | {r['atlas']} | {cval} | {_f(r['auc']['mean'])} ± {_f(r['auc']['std'])} | "
                     f"{_f(r['accuracy']['mean'])} | {_f(r['balanced_accuracy']['mean'])} | "
                     f"{_f(r['sensitivity']['mean'])} | {_f(r['specificity']['mean'])} |")
    lines += ["", f"Noise-aware ensemble against each baseline on the {len(kf['folds'])} shared splits "
              "(corrected resampled t-test, one-sided):", "",
              "| Baseline | ΔAUC | p | splits won / lost | ΔBalanced acc. | p |", "|---|---|---|---|---|---|"]
    comps = {**{k: v for k, v in kf["comparisons"].items() if k.endswith("|ensemble")}, **deep_comp}
    for key, c in comps.items():
        name = DEEP.get(key) or f"{MODELS[key.split('|')[0]]}, ensemble"
        a, b = c["auc"], c["balanced_accuracy"]
        lines.append(f"| {name} | {_d(a['mean_diff'])} | {_p(a['p_one_sided'])} | {a['wins']} / {a['losses']} | "
                     f"{_d(b['mean_diff'])} | {_p(b['p_one_sided'])} |")
    lo = study.get("loso")
    if lo:
        lines += ["", f"### Leave one site out ({len(lo['folds'])} held-out sites)", "",
                  "| Model | AUC | Accuracy | Balanced acc. |", "|---|---|---|---|"]
        for r in sorted((r for r in lo["rows"] if r["atlas"] == "ensemble"), key=lambda r: order.index(r["model"])):
            lines.append(f"| {r['name']}, ensemble | {_f(r['auc']['mean'])} | {_f(r['accuracy']['mean'])} | "
                         f"{_f(r['balanced_accuracy']['mean'])} |")
    return lines + [""]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(prog="brainssl report", description=__doc__.split("\n\n")[0])
    parser.add_argument("--smri-runs", type=Path, default=Path("runs/smri"))
    parser.add_argument("--study", type=Path, default=Path("runs/fmri_study"))
    parser.add_argument("--deep-runs", type=Path, default=Path("runs/fmri"))
    parser.add_argument("--choices", type=Path, default=Path("assets/fmri_choices.json"))
    parser.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    smri = load_smri_runs(args.smri_runs) if args.smri_runs.exists() else pd.DataFrame()
    smri_cmp = smri_comparisons(smri) if not smri.empty else []

    study = build(args.study, args.choices) if (args.study / "kfold" / "labels.npy").exists() else {}
    deep_table, deep_comp = [], {}
    if study.get("kfold") and args.deep_runs.exists():
        settings = read_json(args.study / "settings_kfold.json")
        deep = {}
        for key, run in load_fmri_runs(args.deep_runs).items():
            if same_splits(args.deep_runs / key, settings):
                deep[key] = run
            else:
                log.warning("%s used other splits than the study; left out of the comparison", key)
        kf = study["kfold"]
        labels = np.array(kf["labels"])
        ours = np.array(kf["ensemble_probs"]["vicinal"])
        deep_table, deep_comp = deep_rows(deep, labels, kf["folds"], ours, len(labels))

    lines = ["# Results", "", "Generated by `brainssl report` from the run directories; do not edit by hand.", ""]
    lines += smri_section(smri, smri_cmp) + fmri_section(study, deep_table, deep_comp)
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("ensemble_probs", "labels", "folds")}
            if isinstance(v, dict) else v for k, v in study.items()}
    results = {"smri": {"runs": smri.to_dict(orient="records") if not smri.empty else [], "comparisons": smri_cmp},
               "fmri_study": slim, "fmri_deep": {"rows": deep_table, "comparisons": deep_comp}}
    write_json(args.out / "results.json", results)
    log.info("report written to %s", args.out)
    return results
