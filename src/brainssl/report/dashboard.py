"""``brainssl dashboard``: the data files behind the dashboard (dashboard/data/).

Everything the page shows is computed here from run artefacts: out-of-fold
predictions, per-case test scores, saved probability maps and the exported
models. Nothing is typed in by hand.

    brainssl dashboard --smri-runs runs/smri --study runs/fmri_study --deep-runs runs/fmri \
        --exports exports --out dashboard/data

Showcase scans for the viewer are fixed in assets/showcase_cases.json by a rule
on reference tumour volume alone (the median scan of each volume quartile of
the test set), not by how well any model did on them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from ..smri.infer import enforce_hierarchy
from ..smri.preprocess import _take, labels_from_regions, read_label, read_volume
from ..stats import auc_score, bootstrap_ci
from ..utils import read_json, write_json
from .cli import same_splits
from .collect import calibration_curve, load_fmri_runs, load_smri_runs
from .fmri_study import MODELS, compare, ensemble, load_study, select

log = logging.getLogger(__name__)

REGIONS = ("WT", "TC", "ET")
ATLAS_LABELS = {"cc200": "CC200 (200 regions)", "cc400": "CC400 (392 regions)", "aal": "AAL (116 regions)",
                "ho": "Harvard-Oxford (111 regions)", "dosenbach160": "Dosenbach (161 regions)"}


# --------------------------------------------------------------------------- helpers

def mean_ci(values: np.ndarray) -> dict:
    point, lo, hi = bootstrap_ci(np.mean, np.asarray(values, float))
    return {"mean": point, "lo": lo, "hi": hi}


def roi_names(raw: Path, atlas: str, labels: np.ndarray) -> list[str]:
    """Readable region names from the label files of the atlas release."""
    folder = raw / "abide" / "atlases"
    if atlas in ("cc200", "cc400"):
        table = pd.read_csv(folder / ("CC200_ROI_labels.csv" if atlas == "cc200" else "CC400_ROI_labels.csv"))
        ho = table.columns[-1]
        names = {}
        for _, row in table.iterrows():
            best = str(row[ho]).split("]")[0].strip('["').split('":')[0]
            names[int(row.iloc[0])] = best if best and best != "None" else f"region {int(row.iloc[0])}"
        return [names.get(int(l), f"region {int(l)}") for l in labels]
    files = {"aal": "aal_labels.csv", "ho": "ho_labels.csv", "dosenbach160": "dos160_labels.csv"}
    mapping = {}
    for line in (folder / files[atlas]).read_text().splitlines():
        if line and line[0].isdigit():
            k, v = line.split(",", 1)
            mapping[int(k)] = v.strip()
    return [mapping.get(int(l), f"region {int(l)}") for l in labels]


def vec_index_pairs(n: int) -> np.ndarray:
    iu = np.triu_indices(n)
    return np.stack(iu, 1)


# --------------------------------------------------------------------------- fMRI

ATLAS_ORDER = ("cc200", "cc400", "aal", "ho", "dosenbach160")
DEEP_LABELS = {"mlp_cc200_kfold": "MLP", "brainnetcnn_cc200_kfold": "BrainNetCNN",
               "bnt_cc200_kfold": "Brain Network Transformer"}


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p >= 0.5) == y).mean())


def _balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    if y.min() == y.max():
        raise ValueError("balanced accuracy needs both classes")
    pred = p >= 0.5
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


METRICS = {"auc": auc_score, "accuracy": _accuracy, "balanced_accuracy": _balanced_accuracy}


def score_row(labels: np.ndarray, prob: np.ndarray, n_boot: int = 1000) -> dict:
    """Mean and spread over repeats, and a bootstrap interval over subjects for that mean."""
    out = {}
    for name, fn in METRICS.items():
        per_repeat = np.array([fn(labels, p) for p in prob])

        def over_repeats(y, *ps, fn=fn):
            return float(np.mean([fn(y, p) for p in ps]))
        _, lo, hi = bootstrap_ci(over_repeats, labels, *prob, n_boot=n_boot)
        out[name] = {"mean": float(per_repeat.mean()), "std": float(per_repeat.std()), "lo": lo, "hi": hi}
    return out


def build_fmri(study: Path, deep_runs: Path, choices_path: Path, out: Path) -> dict | None:
    """Per atlas and for the atlas ensemble: every model, the strongest baseline, and the paired tests."""
    if not (study / "kfold" / "labels.npy").exists():
        return None
    choices = read_json(choices_path)
    kf = load_study(study, "kfold")
    labels, folds, n = kf["labels"], kf["folds"], len(kf["labels"])

    def picks(st: dict) -> dict:
        return {m: select(st, m, choices["vicinal_C"] if m == "vicinal" else None) for m in st["probs"]}

    chosen = picks(kf)
    atlases = [a for a in ATLAS_ORDER if a in chosen["vicinal"]]
    candidates = {"ensemble": {m: (None, ensemble(sel)) for m, sel in chosen.items()}}
    for atlas in atlases:
        candidates[atlas] = {m: sel[atlas] for m, sel in chosen.items() if atlas in sel}
    if deep_runs.exists():
        settings = read_json(study / "settings_kfold.json")
        for key, run in load_fmri_runs(deep_runs).items():
            if key in DEEP_LABELS and same_splits(deep_runs / key, settings):
                for view in ("ensemble", "cc200"):
                    if view in candidates:
                        candidates[view][key] = (None, run["oof"]["prob"])

    views = []
    for view_id, models in candidates.items():
        rows = []
        for key, (C, prob) in models.items():
            label = MODELS.get(key) or f"{DEEP_LABELS[key]} (cc200)"
            rows.append({"key": key, "label": label, "ours": key == "vicinal", "C": C, **score_row(labels, prob)})
        rows.sort(key=lambda r: -r["auc"]["mean"])
        main = max((r for r in rows if not r["ours"]), key=lambda r: r["auc"]["mean"])
        cmp = compare(labels, folds, models["vicinal"][1], models[main["key"]][1], n)
        tests = {m: {"mean_diff": c["mean_diff"], "p_one_sided": c["p_one_sided"], "wins": c["wins"],
                     "losses": c["losses"],
                     "fold_diffs": [{"id": i, "value": v} for i, v in zip(c["fold_ids"], c["fold_diffs"])]}
                 for m, c in cmp.items() if m in METRICS}
        views.append({"id": view_id, "label": "All five atlases" if view_id == "ensemble" else ATLAS_LABELS[view_id],
                      "rows": rows, "main": main["key"], "main_label": main["label"], "tests": tests})

    ens = views[0]
    main_linear = max((r for r in ens["rows"] if r["key"] in MODELS and not r["ours"]), key=lambda r: r["auc"]["mean"])
    calibration = {}
    for key in ("vicinal", main_linear["key"]):
        pooled = np.nanmean(candidates["ensemble"][key][1], 0)
        calibration["ours" if key == "vicinal" else "baseline"] = [
            {"x": c["mean_prob"], "y": c["frac_positive"], "n": c["n"]} for c in calibration_curve(labels, pooled)]

    sites = loso = None
    if (study / "loso" / "labels.npy").exists():
        lo = load_study(study, "loso")
        lo_ens = {m: ensemble(sel) for m, sel in picks(lo).items()}
        loso = [{"key": m, "label": MODELS[m], "ours": m == "vicinal", **score_row(lo["labels"], p)}
                for m, p in lo_ens.items()]
        loso.sort(key=lambda r: -r["auc"]["mean"])
        site_ids = lo["sites"]
        sites = []
        for site in np.unique(site_ids):
            mask = site_ids == site
            y = lo["labels"][mask]
            sites.append({"site": str(site), "n": int(mask.sum()),
                          "ours": _accuracy(y, lo_ens["vicinal"][0][mask]),
                          "baseline": _accuracy(y, lo_ens[main_linear["key"]][0][mask])})
        sites.sort(key=lambda r: -r["n"])

    repeats = 1 + max(f["repeat"] for f in folds)
    data = {"subjects": int(n), "sites_total": int(len(np.unique(kf["sites"]))), "repeats": repeats,
            "folds": len(folds) // repeats, "views": views, "calibration": calibration,
            "calibration_baseline": main_linear["label"], "sites": sites, "loso": loso,
            "site_baseline": main_linear["label"], "choices": choices["vicinal_C"]}
    write_json(out / "fmri.json", data)
    return data


def build_connectome(exports: Path, raw: Path, processed: Path, out: Path, top: int = 400) -> None:
    meta = json.loads((exports / "connectome" / "connectome.json").read_text())
    (out / "connectome").mkdir(parents=True, exist_ok=True)
    index = []
    for atlas, spec in meta["atlases"].items():
        params = np.fromfile(exports / "connectome" / spec["file"], dtype="<f4")
        d = spec["n_features"]
        w = params[2 * d: 3 * d]
        n = spec["n_rois"]
        pairs = vec_index_pairs(n)
        off = pairs[:, 0] != pairs[:, 1]
        order = np.argsort(-np.abs(w[off]))[:top]
        sel = pairs[off][order]
        edges = [[int(i), int(j), round(float(v), 5)] for (i, j), v in zip(sel, w[off][order])]
        labels = np.array(spec["roi_labels"])
        write_json(out / "connectome" / f"{atlas}.json", {
            "atlas": ATLAS_LABELS.get(atlas, atlas), "n_rois": n, "n_edges": int(off.sum()),
            "names": roi_names(raw, atlas, labels), "xyz": spec["centroids_mm"], "edges": edges})
        index.append({"id": atlas, "label": ATLAS_LABELS.get(atlas, atlas)})
    write_json(out / "connectome" / "index.json", {"atlases": index})


def brain_meshes(out: Path) -> None:
    import nilearn

    src = Path(nilearn.__file__).parent / "datasets" / "data" / "fsaverage5"
    (out / "brain").mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        with gzip.open(src / f"pial_{side}.gii.gz", "rb") as fh:
            (out / "brain" / f"pial_{side}.gii").write_bytes(fh.read())


# --------------------------------------------------------------------------- segmentation

def scans_to_match(target: float, points: list[tuple[int, float]]) -> float | None:
    """Scans the baseline needs to reach ``target``, interpolating its curve linearly in log(scans).

    None when the target lies outside the range the baseline curve covers.
    """
    for (n0, d0), (n1, d1) in zip(points, points[1:]):
        if d0 <= target <= d1 and d1 > d0:
            t = (target - d0) / (d1 - d0)
            return float(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0))))
    return None


def build_smri(smri_runs: Path, out: Path, splits: Path) -> dict | None:
    """Label-efficiency curves, per-scan pairs, the ablation and HD95, for budgets where both models finished."""
    runs = load_smri_runs(smri_runs)
    if runs.empty:
        return None
    sizes = {k: len(v) for k, v in read_json(splits)["budgets"].items()}
    curve = {"ours": [], "baseline": []}
    per_case = {}
    hd = []
    for pct in sorted(runs.budget_pct.unique()):
        group = runs[runs.budget_pct == pct]
        pair = {}
        for key, pattern in (("ours", "ours"), ("baseline", "unet_scratch")):
            hit = group[group.tag.str.contains(pattern)]
            if not hit.empty:
                pair[key] = pd.read_csv(Path(hit.iloc[0]["dir"]) / "eval_test.csv").set_index("case_id")
        if len(pair) < 2:
            continue
        for key, cases in pair.items():
            point = {"budget": f"{pct}%", "scans": sizes.get(f"{pct}pct")}
            for m in ["dice_mean"] + [f"dice_{r}" for r in REGIONS]:
                point[m] = mean_ci(cases[m].to_numpy())
            curve[key].append(point)
            hd.append({"budget": f"{pct}%", "model": "Bilateral U-Net, pretrained" if key == "ours" else "U-Net from scratch",
                       "ours": key == "ours", **{r: float(cases[f"hd95_{r}"].mean()) for r in REGIONS}})
        common = pair["ours"].index.intersection(pair["baseline"].index)
        per_case[f"{pct}%"] = [{"case_id": c, "ours": {m: float(pair["ours"].loc[c, m]) for m in pair["ours"].columns},
                                "baseline": {m: float(pair["baseline"].loc[c, m]) for m in pair["baseline"].columns}}
                               for c in common]
    if not per_case:
        return None
    budgets = [p["budget"] for p in curve["ours"]]
    ablation = []
    ten = runs[runs.budget_pct == 10]
    labels = {"unet_scratch": ("U-Net from scratch", False, True), "unet_mim": ("U-Net, random masked pretraining", False, False),
              "bilateral_scratch": ("Bilateral U-Net from scratch", False, False), "ours": ("Bilateral U-Net, contralateral pretraining", True, False)}
    for pattern, (label, is_ours, is_base) in labels.items():
        hit = ten[ten.tag.str.contains(pattern)]
        if hit.empty:
            continue
        cases = pd.read_csv(Path(hit.iloc[0]["dir"]) / "eval_test.csv")
        ci = mean_ci(cases["dice_mean"].to_numpy())
        ablation.append({"label": label, "ours": is_ours, "baseline": is_base, **ci})
    data = {"budgets": budgets, "default_budget": "10%" if "10%" in budgets else budgets[0],
            "test_cases": int(len(next(iter(per_case.values())))),
            "curve": curve, "per_case": per_case, "ablation": ablation, "hd95": hd}
    write_json(out / "smri.json", data)
    return data


def quantise(volume: np.ndarray, brain: np.ndarray) -> np.ndarray:
    values = volume[brain]
    lo, hi = np.percentile(values, [0.5, 99.5]) if values.size else (0, 1)
    scaled = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1) * 254 + 1
    return np.where(brain, scaled, 0).astype(np.uint8)


def save_volume(folder: Path, name: str, volume: np.ndarray) -> str:
    """8-bit NIfTI with 1 mm voxels; returns the path the page fetches it from."""
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), np.eye(4)), folder / f"{name}.nii.gz")
    return f"data/cases/{folder.name}/{name}.nii.gz"


def run_file(smri_runs: Path, tag: str, name: str) -> Path:
    """A file of run ``tag``: in the run directory, or in ``<tag>_eval`` where the pipeline scores runs."""
    direct = smri_runs / tag / name
    return direct if direct.exists() else smri_runs / f"{tag}_eval" / name


def build_cases(smri_runs: Path, out: Path, raw: Path, index_csv: Path, cases_json: Path,
                budget: str = "10pct") -> None:
    """Viewer volumes for the showcase scans, on the grid the network saw.

    The scans are listed in assets/showcase_cases.json, picked by tumour volume
    alone; the probability maps come from ``brainssl smri evaluate --save-cases``.
    """
    index = pd.read_csv(index_csv).set_index("case_id")
    chosen = read_json(cases_json)
    runs = {"ours": f"ft_ours_{budget}", "baseline": f"ft_unet_scratch_{budget}"}
    scores = {k: pd.read_csv(run_file(smri_runs, tag, "eval_test.csv")).set_index("case_id")
              for k, tag in runs.items()}
    cfg = read_json(smri_runs / runs["ours"] / "config.json")
    window = [cfg.get("lr_width", 160), *cfg.get("patch", (128, 128))]
    root = out / "cases"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for case_id in chosen["cases"]:
        row = index.loc[case_id]
        saved = {k: np.load(run_file(smri_runs, tag, f"predictions/{case_id}.npz")) for k, tag in runs.items()}
        box = tuple(slice(int(a), int(b)) for a, b in saved["ours"]["box"])
        image = _take(read_volume(raw / row["image"]), box)
        brain = (image > 0).any(axis=0)
        truth = _take(read_label(raw / row["label"]), box)
        folder = root / case_id
        folder.mkdir(exist_ok=True)
        files = {name: save_volume(folder, name.lower(), quantise(image[c], brain))
                 for c, name in enumerate(("FLAIR", "T1", "T1Gd", "T2"))}
        files["truth"] = save_volume(folder, "truth", truth)
        preds = {k: labels_from_regions(enforce_hierarchy(v["prob"].astype(np.float32))) for k, v in saved.items()}
        for key, label in preds.items():
            files[key] = save_volume(folder, key, label)
        errors = np.zeros_like(truth)
        errors[(truth > 0) & (preds["ours"] == 0)] = 1      # missed tumour
        errors[(truth == 0) & (preds["ours"] > 0)] = 2      # false tumour
        files["errors"] = save_volume(folder, "errors", errors)
        flair = image[0]
        asym = np.abs(flair - flair[::-1]) / (np.percentile(flair[brain], 99) + 1e-6)
        files["asymmetry"] = save_volume(folder, "asymmetry", np.clip(asym, 0, 1) * 255 * brain)
        volume_ml = float(row[["n_label1", "n_label2", "n_label3"]].sum()) / 1000
        entries.append({
            "id": case_id, "files": files, "window": window, "tag": f"{volume_ml:.0f} ml tumour",
            "dice": {k: {r: float(df.loc[case_id, f"dice_{r}"]) for r in REGIONS} for k, df in scores.items()},
            "hd95": {k: {r: float(df.loc[case_id, f"hd95_{r}"]) for r in REGIONS} for k, df in scores.items()},
            "note": "Chosen as the median-volume test scan of one quartile of tumour volume, "
                    "before any model was scored on it.",
        })
    write_json(root / "index.json", {"cases": entries, "hero": entries[len(entries) // 2]["id"],
                                     "rule": chosen["rule"]})


# --------------------------------------------------------------------------- fMRI demo

def build_fmri_demo(demo_dir: Path, out: Path) -> None:
    """Copy the fold-specific demo export written by `brainssl export demo`."""
    target = out / "demo" / "fmri"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(demo_dir, target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brainssl dashboard", description=__doc__.split("\n\n")[0])
    parser.add_argument("--smri-runs", type=Path, default=Path("runs/smri"))
    parser.add_argument("--study", type=Path, default=Path("runs/fmri_study"))
    parser.add_argument("--deep-runs", type=Path, default=Path("runs/fmri"))
    parser.add_argument("--choices", type=Path, default=Path("assets/fmri_choices.json"))
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    parser.add_argument("--showcase-cases", type=Path, default=Path("assets/showcase_cases.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("dashboard/data"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    brain_meshes(args.out)
    summary: dict = {"environment": {
        "GPU": "NVIDIA GeForce GTX 1080 Ti, 11 GB", "CPU": "Intel Xeon E5-2697 v4, 18 cores", "RAM": "32 GB",
        "OS": "Windows 10", "Python": "3.11.8", "PyTorch": "2.7.1 + CUDA 12.6"}}
    fmri = build_fmri(args.study, args.deep_runs, args.choices, args.out)
    if fmri:
        ens = fmri["views"][0]
        ours = next(r for r in ens["rows"] if r["ours"])
        main_row = next(r for r in ens["rows"] if r["key"] == ens["main"])
        t = ens["tests"]["auc"]
        summary["fmri"] = {"ours": ours["auc"]["mean"], "baseline": main_row["auc"]["mean"],
                           "baseline_name": ens["main_label"], "wins": t["wins"], "splits": len(t["fold_diffs"]),
                           "p": t["p_one_sided"], "repeats": fmri["repeats"], "folds": fmri["folds"],
                           "subjects": fmri["subjects"],
                           "ranking": [{"label": r["label"], "value": r["auc"]["mean"], "lo": r["auc"]["lo"],
                                        "hi": r["auc"]["hi"], "ours": r["ours"], "main": r["key"] == ens["main"]}
                                       for r in ens["rows"]]}
    if (args.exports / "connectome" / "connectome.json").exists():
        build_connectome(args.exports, args.raw, args.processed, args.out)
    if (args.exports / "demo_fmri" / "index.json").exists():
        build_fmri_demo(args.exports / "demo_fmri", args.out)
    splits = args.processed / "msd" / "splits.json"
    smri = build_smri(args.smri_runs, args.out, splits) if args.smri_runs.exists() and splits.exists() else None
    if smri:
        b = smri["default_budget"]
        o = next(p for p in smri["curve"]["ours"] if p["budget"] == b)
        s = next(p for p in smri["curve"]["baseline"] if p["budget"] == b)
        base_curve = [(p["scans"], p["dice_mean"]["mean"]) for p in smri["curve"]["baseline"]]
        match = scans_to_match(o["dice_mean"]["mean"], base_curve)
        summary["smri"] = {"headline_budget": b, "ours": o["dice_mean"]["mean"], "baseline": s["dice_mean"]["mean"],
                           "test_cases": smri["test_cases"], "headline_scans": o["scans"],
                           "baseline_scans_to_match": match,
                           "curve": {"ticks": smri["budgets"],
                                     "ours": [{"x": p["budget"], "y": p["dice_mean"]["mean"], "lo": p["dice_mean"]["lo"], "hi": p["dice_mean"]["hi"]} for p in smri["curve"]["ours"]],
                                     "baseline": [{"x": p["budget"], "y": p["dice_mean"]["mean"], "lo": p["dice_mean"]["lo"], "hi": p["dice_mean"]["hi"]} for p in smri["curve"]["baseline"]]}}
        first = read_json(args.showcase_cases)["cases"][0]
        if run_file(args.smri_runs, "ft_ours_10pct", f"predictions/{first}.npz").exists():
            build_cases(args.smri_runs, args.out, args.raw, args.processed / "msd" / "index.csv", args.showcase_cases)
    write_json(args.out / "summary.json", summary)
    log.info("dashboard data written to %s", args.out)
