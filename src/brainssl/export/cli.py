"""``brainssl export``: models and bundles that leave the repository.

    brainssl export segmentation --weights runs/smri/ft_ours_100pct/best.pt --out exports/segmentation --bilateral
    brainssl export connectome --out exports/connectome
    brainssl export demo-fmri --out exports/demo_fmri
    brainssl export all --out exports          # all of the above plus model cards

The segmentation folder and the connectome folder are what goes to the Hugging
Face Hub (scripts/push_to_hub.py); the demo bundle feeds the dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from ..utils import write_json

log = logging.getLogger(__name__)
DEMO_ATLASES = ("cc200", "aal", "ho")


def export_segmentation_bundle(run: Path, out: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    from .onnx_export import export_segmentation

    cfg = json.loads((run / "config.json").read_text())
    weights = run / "best.pt"
    export_segmentation(weights, out, bilateral=cfg["bilateral"], channels=tuple(cfg["channels"]))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in state.items() if not k.startswith(("extra.", "aux."))}
    save_file(state, str(out / "model.safetensors"))
    write_json(out / "config.json", {"architecture": "UNet3D", "in_channels": 4, "out_channels": 3,
                                     "channels": cfg["channels"], "bilateral": cfg["bilateral"],
                                     "bilateral_levels": [1, 2, 3, 4], "regions": ["WT", "TC", "ET"],
                                     "trained_on": cfg.get("budget"), "source_run": run.name})
    return out


def export_connectome_bundle(out: Path, raw: Path, processed: Path, choices: Path) -> Path:
    from ..fmri.final import export_connectome

    chosen = json.loads(choices.read_text())["vicinal_C"]
    export_connectome(chosen, out, raw, processed, n_windows=6, fraction=0.7)
    return out


def export_demo_fmri(out: Path, raw: Path, processed: Path, choices: Path, n_subjects: int = 16,
                     folds: int = 10, seed: int = 10) -> Path:
    """A model fitted on the training subjects of one reported split, and that
    split's held-out subjects, so the browser demo only scores unseen scans."""
    from ..fmri.abide import load_cache, load_phenotypes
    from ..fmri.connectivity import sym_to_vec
    from ..fmri.evaluate import kfold_splits
    from ..fmri.final import fit_atlas
    from ..fmri.methods import ConnectomeData
    from ..fmri.study import _sigmoid

    chosen = json.loads(choices.read_text())["vicinal_C"]
    atlases = [a for a in DEMO_ATLASES if a in chosen] or list(chosen)[:1]
    pheno = load_phenotypes(raw / "abide")
    fold = kfold_splits(pheno["label"].to_numpy(), pheno["site"].to_numpy(), folds, 1, seed)[0]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    test = fold.test
    by_label = [test[pheno["label"].to_numpy()[test] == lab] for lab in (0, 1)]
    per_label = min(n_subjects // 2, *(len(g) for g in by_label))
    picks = np.concatenate([rng.choice(g, per_label, replace=False) for g in by_label])
    subjects = {int(i): {"id": str(pheno["FILE_ID"].iloc[i]), "site": str(pheno["site"].iloc[i]),
                         "label": int(pheno["label"].iloc[i]), "atlases": {}} for i in picks}
    model_meta: dict = {"windows": {"count": 6, "fraction": 0.7, "min_length": 40}, "atlases": {},
                        "split": "repeat 1, fold 1 of the reported cross-validation"}
    for atlas in atlases:
        data = ConnectomeData.load(raw, processed, (atlas,), windows=True)
        sub = ConnectomeData(data.labels[fold.train], data.sites[fold.train], (atlas,),
                             {atlas: data.corr[atlas][fold.train]}, {atlas: data.logs[atlas][fold.train]},
                             {atlas: data.windows[atlas][fold.train]})
        params = fit_atlas(sub, atlas, chosen[atlas], vicinal=True)
        bias = params["bias"]

        def z(x, p=params):
            return (x - p["mean"]) / p["std"]
        blob = np.concatenate([params["mean"], params["std"], params["weight"]]).astype("<f4")
        blob.tofile(out / f"model_{atlas}.bin")
        model_meta["atlases"][atlas] = {"file": f"model_{atlas}.bin", "bias": bias, "n_features": len(params["weight"])}
        ts_all = load_cache(processed / "abide", atlas).subset(pheno["FILE_ID"].tolist())
        for i in picks:
            ts = ts_all[i].astype("<f4")
            name = f"{subjects[int(i)]['id']}_{atlas}.bin"
            ts.tofile(out / name)
            s_full = z(sym_to_vec(data.logs[atlas][i:i + 1])) @ params["weight"]
            s_win = z(data.windows[atlas][i]) @ params["weight"]
            p = 0.5 * (_sigmoid(s_full + bias) + _sigmoid(s_win + bias).mean())
            subjects[int(i)]["atlases"][atlas] = {"file": name, "T": int(ts.shape[0]), "R": int(ts.shape[1]),
                                                  "python_probability": float(np.ravel(p)[0])}
    write_json(out / "index.json", {"model": model_meta, "subjects": list(subjects.values())})
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="brainssl export", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="what", required=True)
    s = sub.add_parser("segmentation")
    s.add_argument("--run", type=Path, required=True, help="fine-tuning run directory")
    s.add_argument("--out", type=Path, default=Path("exports/segmentation"))
    for name in ("connectome", "demo-fmri", "all"):
        q = sub.add_parser(name)
        default_out = Path("exports") if name == "all" else Path("exports") / name.replace("-", "_")
        q.add_argument("--out", type=Path, default=default_out)
        q.add_argument("--raw", type=Path, default=Path("data/raw"))
        q.add_argument("--processed", type=Path, default=Path("data/processed"))
        q.add_argument("--choices", type=Path, default=Path("assets/fmri_choices.json"))
        if name in ("demo-fmri", "all"):
            q.add_argument("--folds", type=int, default=10, help="folds of the reported cross-validation")
        if name == "all":
            q.add_argument("--smri-run", type=Path, default=Path("runs/smri/ft_ours_100pct"))
            q.add_argument("--report", type=Path, default=Path("report/results.json"))
    args = p.parse_args(argv)
    if args.what == "segmentation":
        export_segmentation_bundle(args.run, args.out)
    elif args.what == "connectome":
        export_connectome_bundle(args.out, args.raw, args.processed, args.choices)
    elif args.what == "demo-fmri":
        export_demo_fmri(args.out, args.raw, args.processed, args.choices, folds=args.folds)
    else:
        from .cards import write_cards

        seg = export_segmentation_bundle(args.smri_run, args.out / "segmentation")
        con = export_connectome_bundle(args.out / "connectome", args.raw, args.processed, args.choices)
        export_demo_fmri(args.out / "demo_fmri", args.raw, args.processed, args.choices, folds=args.folds)
        results = json.loads(args.report.read_text()) if args.report.exists() else {}
        write_cards(seg, con, results)
        shutil.copy(args.choices, args.out / "fmri_choices.json")
        log.info("exports written to %s", args.out)
