"""``brainssl trust report``: the trust results as a Markdown page, from the audit files.

Every number on the page is read from ``trust_fmri.json``, ``trust_interpret.json``
and ``trust_smri.json``; nothing is typed in. Tables are ordered the same way
for every model, and results that go against the proposed model are printed in
the same way as the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_NAMES = {"vicinal": "Noise-aware (ours)", "tangent": "Tangent space + LR", "log": "Log-Euclidean + LR",
               "corr": "Correlation + LR"}
ORDER = ("vicinal", "tangent", "log", "corr")
REGIONS = ("WT", "TC", "ET")


def _f(x, d: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ms(block: dict, d: int = 3) -> str:
    return f"{_f(block['mean'], d)} ± {_f(block['std'], d)}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + body + [""]


def _fmri(data: dict) -> list[str]:
    lines = ["## Connectome classifier (ABIDE I)", ""]
    for scheme, title in (("kfold", "5 × 10-fold cross-validation"), ("loso", "leave one site out")):
        block = data.get(scheme)
        if not block:
            continue
        lines += [f"### Calibration, {title}", "",
                  "Raw probabilities, and the same probabilities after a monotone recalibration fitted on "
                  "other folds only. Slope 1 is calibrated; above 1 the predictions are too timid.", ""]
        rows = []
        for m in ORDER:
            if m not in block["recalibration"]:
                continue
            r = block["recalibration"][m]
            rows.append([MODEL_NAMES[m], _ms(r["raw"]["ece"]), _ms(r["raw"]["slope"], 2), _ms(r["raw"]["brier"]),
                         _ms(r["recalibrated"]["ece"]), _ms(r["recalibrated"]["slope"], 2),
                         _ms(r["recalibrated"]["brier"]), _f(r["raw"]["balanced_accuracy"]["mean"]),
                         _f(r["recalibrated"]["balanced_accuracy"]["mean"])])
        lines += _table(["Model", "ECE", "slope", "Brier", "ECE (recal.)", "slope (recal.)", "Brier (recal.)",
                         "bal. acc.", "bal. acc. (recal.)"], rows)
        lines += [f"### Selective prediction, {title}", "",
                  "Accuracy when only the most confident fraction of subjects is scored. `excess` is 0 for an "
                  "ideal ranking of cases by confidence and 1 for a ranking no better than chance.", ""]
        rows = []
        for m in ORDER:
            if m not in block["selective"]:
                continue
            s = block["selective"][m]
            rows.append([MODEL_NAMES[m], _f(s["accuracy_full"]), _f(s["accuracy_at_80"]), _f(s["accuracy_at_50"]),
                         _ms(s["excess"], 2)])
        lines += _table(["Model", "all subjects", "most confident 80%", "most confident 50%", "excess"], rows)
    kf = data.get("kfold", {})
    if "shortcuts" in kf:
        sc = kf["shortcuts"]
        lines += ["### Is the AUC a shortcut?", "",
                  "Classifiers that see only nuisance variables, on the same folds, against the connectome model.", ""]
        rows = [[f"only {r['inputs']}", _ms(r["auc"])] for r in sc["nuisance_only"]]
        m = sc["model"]
        rows += [["connectome model, all subjects", _ms(m["auc"])],
                 ["connectome model, inside sites", _ms(m["auc_within_site"])],
                 ["connectome model, inside motion terciles", _ms(m["auc_within_motion_tercile"])],
                 ["connectome model, inside site and motion", _ms(m["auc_within_site_and_motion"])]]
        lines += _table(["Inputs", "AUC"], rows)
        def rho(r: dict) -> str:
            return f"{_f(r['rho'], 2)} [{_f(r['ci_low'], 2)}, {_f(r['ci_high'], 2)}]"

        lines += ["Rank correlation of the model score with mean framewise displacement: "
                  f"controls {rho(m['score_vs_motion_controls'])}, autism {rho(m['score_vs_motion_autism'])}.", ""]
        sub = kf["subgroups"]["vicinal"]
        s = sub["site_summary"]
        lines += ["### Subgroups", "",
                  f"AUC of the noise-aware model inside each group (mean of the five repeats' out-of-fold "
                  f"probabilities). {s['sites_scored']} of {s['sites_total']} sites have enough subjects in both "
                  f"classes to be scored; the interval excludes 0.5 for {s['sites_ci_above_half']} of them. "
                  f"Worst site {_f(s['worst_auc'])}, median {_f(s['median_auc'])}, best {_f(s['best_auc'])}.", ""]
        for kind in ("site", "sex", "age", "motion"):
            rows = [[r["group"], str(r["n"]), str(r["positives"]),
                     "too few" if r["auc"] is None else f"{_f(r['auc'])} [{_f(r['ci_low'])}, {_f(r['ci_high'])}]"]
                    for r in sub[kind]]
            lines += _table([kind, "subjects", "autism", "AUC [95% CI]"], rows)
    return lines


def _interpret(data: dict) -> list[str]:
    lines = ["### What the connectome classifier looks at", "",
             "Weights are on log-connectivity features, so single entries are not read one by one. Strongest "
             "regions by summed absolute weight (node strength):", ""]
    for atlas, w in data["weights"].items():
        top = ", ".join(f"{r['name']} ({_f(r['strength'], 2)})" for r in w["top_rois"][:6])
        lines.append(f"- **{atlas}**: {top}")
    lines.append("")
    aal = data["weights"].get("aal")
    if aal:
        names = aal["blocks"]["names"]
        mat = np.array(aal["blocks"]["mean_abs"])
        pairs = sorted(((mat[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i, len(names))),
                       reverse=True)[:8]
        lines += ["Lobe blocks of the AAL atlas with the largest mean absolute weight per feature:", ""]
        lines += _table(["block", "mean abs. weight"], [[f"{a} – {b}", _f(v, 4)] for v, a, b in pairs])
    if data.get("stability"):
        lines += ["#### Stability under bootstrap refits", "",
                  "Refits on resamples of the subjects. `top-100 overlap` is the Jaccard overlap of the 100 "
                  "strongest features between two refits; node strengths summarise the weights by region.", ""]
        rows = [[a, str(s["bootstraps"]), _f(s["top_k_jaccard_pairs"], 2), _f(s["weight_rank_correlation"], 2),
                 _f(s["node_strength_rank_correlation"], 2), _f(s["node_strength_vs_full"], 2),
                 _f(s["top_features_sign_agreement"], 2)] for a, s in data["stability"].items()]
        lines += _table(["atlas", "refits", "top-100 overlap", "weight rank corr.", "node strength rank corr.",
                         "node strength vs full fit", "sign agreement, top 100"], rows)
    if data.get("deletion"):
        lines += ["#### Faithfulness: deleting the features an explanation names", "",
                  "Held-out AUC after setting a fraction of each subject's features to the training mean. "
                  "The most supportive features are those pushing most towards the predicted class.", ""]
        rows = []
        for atlas, d in data["deletion"].items():
            for frac, r in d["fractions"].items():
                rows.append([atlas, f"{float(frac) * 100:g}%", _f(d["baseline_auc"]), _f(r["supportive"]),
                             _f(r["random"]), _f(r["least_supportive"])])
        lines += _table(["atlas", "deleted", "no deletion", "most supportive", "random", "least supportive"], rows)
    return lines


def _smri(data: dict) -> list[str]:
    runs = data["runs"]
    lines = ["## Tumour segmentation (BraTS)", ""]
    lines += ["### Uncertainty, calibration and volumes", "",
              "Spearman correlation of a scan's mean entropy with its error (1 − Dice); a positive value means the "
              "model is more uncertain on the scans it segments worse. ECE of the voxel probabilities is computed "
              "within five voxels of the predicted or true region.", ""]
    rows = []
    for label, r in runs.items():
        for region in REGIONS:
            u, v = r["uncertainty"][region], r["volume"][region]
            rows.append([label, region, f"{_f(u['entropy']['spearman']['rho'], 2)} "
                         f"[{_f(u['entropy']['spearman']['ci_low'], 2)}, {_f(u['entropy']['spearman']['ci_high'], 2)}]",
                         _f(u["disagree"]["spearman"]["rho"], 2), _f(u["entropy"]["risk_coverage"]["excess"], 2),
                         _f(r["calibration"][region]["ece"], 3), _f(v["bias"], 1), _f(v["relative_error"], 2),
                         _f(v["pearson"], 3)])
    lines += _table(["run", "region", "entropy vs error (ρ)", "disagreement vs error (ρ)", "risk-coverage excess",
                     "voxel ECE", "volume bias (mL)", "median relative volume error", "volume correlation"], rows)
    lines += ["### Dice by tumour size", ""]
    rows = []
    for label, r in runs.items():
        for size, s in r["by_size"].items():
            rows.append([label, f"{size} ({_f(s['wt_range_ml'][0], 0)}–{_f(s['wt_range_ml'][1], 0)} mL)",
                         str(s["cases"])] + [_f(s[f"dice_{n}"]) for n in REGIONS])
    lines += _table(["run", "whole-tumour volume", "scans", "Dice WT", "Dice TC", "Dice ET"], rows)
    lines += ["### Worst scans", ""]
    for label, r in runs.items():
        lines.append(f"**{label}** (enhancing tumour absent in {r['et_absent']['cases']} scans, "
                     f"predicted anyway in {r['et_absent']['false_alarm']})")
        lines.append("")
        lines += _table(["scan", "mean Dice", "what went wrong"],
                        [[w["case"], _f(w["dice_mean"]), "; ".join(w["tags"])] for w in r["worst_cases"][:5]])
    lines += ["### Robustness to worse images", "",
              "Mean Dice over the three regions on the same randomly chosen test scans, with the same perturbed "
              "image given to every model.", ""]
    names = list(next(iter(runs.values()))["robustness"]["perturbations"])
    header = ["perturbation"] + [f"{label} (Δ)" for label in runs]
    rows = [["none"] + [_f(r["robustness"]["clean_mean"]) for r in runs.values()]]
    for name in names:
        cells = []
        for r in runs.values():
            p = r["robustness"]["perturbations"].get(name)
            cells.append(f"{_f(p['mean'])} ({p['drop_mean']:+.3f})" if p else "–")
        rows.append([name] + cells)
    lines += _table(header, rows)
    comps = data.get("robustness_comparisons", {})
    for key, comp in comps.items():
        a, b = key.split("|")
        lines += [f"Paired comparison of the Dice drop, **{a}** against **{b}** (positive: {a} loses less).", ""]
        rows = [[name, f"{c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]", f"{c['wins']}/{c['losses']}",
                 _f(c["wilcoxon_p_one_sided"])] for name, c in comp.items()]
        lines += _table(["perturbation", "difference in drop [95% CI]", "scans better / worse", "p"], rows)
    lines += ["### Which contrasts the models rely on", "",
              "Dice drop when a contrast is removed, and the share of the gradient-times-input attribution inside "
              "the segmented tumour.", ""]
    rows = []
    for label, r in runs.items():
        shares = r.get("saliency_share", {})
        for c in ("FLAIR", "T1", "T1Gd", "T2"):
            drops = r["robustness"]["perturbations"].get(f"no {c}", {}).get("drop", {})
            rows.append([label, c] + [_f(drops.get(n)) for n in REGIONS] + [_f(shares.get(c), 2)])
    lines += _table(["run", "contrast removed", "Dice drop WT", "Dice drop TC", "Dice drop ET",
                     "attribution share"], rows)
    bil = {k: r for k, r in runs.items() if r.get("exchange")}
    if bil:
        lines += ["### The bilateral blocks", "",
                  "`correction` is the size of an exchange block's update relative to its input features, inside and "
                  "outside the tumour; `asymmetry AUC` is how well the left-right asymmetry signal |f − Rf| alone "
                  "separates tumour from healthy tissue. Level 1 is the first downsampled scale.", ""]
        rows = []
        for label, r in bil.items():
            for lv, e in enumerate(r["exchange"], start=1):
                if e:
                    rows.append([label, str(lv), _f(e["correction_in"], 3), _f(e["correction_out"], 3),
                                 _f(e["asymmetry_auc"], 3)])
        lines += _table(["run", "level", "correction inside tumour", "correction outside", "asymmetry AUC"], rows)
        for label, r in bil.items():
            blind = r["robustness"]["perturbations"].get("mirror blind")
            if blind:
                lines.append(f"Dice of **{label}** when the blocks see the scan's own features instead of its mirror "
                             f"image: {_f(blind['mean'])} against {_f(r['robustness']['clean_mean'])} "
                             f"(drop {blind['drop_mean']:+.3f}).")
        lines.append("")
    return lines


def build(directory: Path, out: Path) -> Path:
    parts = {n: directory / f"trust_{n}.json" for n in ("fmri", "interpret", "smri")}
    data = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in parts.items()
            if p.exists() and p.stat().st_size > 2}
    lines = ["# Interpretability and trust: results", "",
             "Generated by `brainssl trust report` from the audit files; the questions and the reading of each "
             "table are on the [plan page](trust.md).", ""]
    if "fmri" in data:
        lines += _fmri(data["fmri"])
    if "interpret" in data:
        lines += _interpret(data["interpret"])
    if "smri" in data:
        lines += _smri(data["smri"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out
