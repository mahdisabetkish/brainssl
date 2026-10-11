"""``brainssl fmri cv``: score one connectome classifier under one protocol.

    brainssl fmri cv --method tangent_cc200 --model linear --embedding tangent --atlases cc200
    brainssl fmri cv --method ours --model linear --embedding log --vicinal --tta \
        --atlases cc200 cc400 aal ho dosenbach160 --scheme loso

Out-of-fold predictions, per-fold metrics and a summary land in
``<runs>/<method>_<scheme>/`` (or ``--out``); finished folds are reused.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..utils import write_json
from .abide import ATLASES
from .evaluate import kfold_splits, run_cv, site_splits
from .methods import ConnectomeData, DeepConnectome, LinearConnectome

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brainssl fmri cv", description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, help="name of the run, e.g. tangent_cc200")
    p.add_argument("--model", choices=["linear", "deep"], default="linear")
    p.add_argument("--embedding", choices=["tangent", "log", "corr"], default="log")
    p.add_argument("--vicinal", action="store_true", help="train on each scan's window connectomes too")
    p.add_argument("--tta", action="store_true", help="average predictions over the scan's windows")
    p.add_argument("--arch", choices=["mlp", "brainnetcnn", "bnt"], default="brainnetcnn")
    p.add_argument("--atlases", nargs="+", default=list(ATLASES))
    p.add_argument("--scheme", choices=["kfold", "loso"], default="kfold")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=10,
                   help="first split seed; development runs used seeds 0-2, reported runs use 10-14")
    p.add_argument("--c", type=float, nargs="+", default=None,
                   help="regularisation grid; one value fixes C, several are searched by inner CV")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--runs", type=Path, default=Path("runs/fmri"))
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    atlases = tuple(args.atlases)
    needs_windows = args.model == "linear" and (args.vicinal or args.tta)
    data = ConnectomeData.load(args.raw, args.processed, atlases, windows=needs_windows)
    if args.scheme == "kfold":
        folds = kfold_splits(data.labels, data.sites, args.folds, args.repeats, args.seed)
    else:
        folds = site_splits(data.sites)

    if args.model == "linear":
        grid = tuple(args.c) if args.c else LinearConnectome.DEFAULT_GRID
        method = LinearConnectome(data, atlases, embedding=args.embedding, vicinal=args.vicinal, tta=args.tta,
                                  c_grid=grid, seed=args.seed, name=args.method)
    else:
        if len(atlases) != 1:
            raise SystemExit("deep baselines are single-atlas; pass one --atlases value")
        method = DeepConnectome(data, atlases[0], args.arch, epochs=args.epochs, seed=args.seed)
        method.name = args.method

    out = args.out or args.runs / f"{args.method}_{args.scheme}"
    # Saved folds are reused on restart, so refuse to mix folds fitted with other settings.
    signature = {k: getattr(args, k) for k in ("model", "embedding", "vicinal", "tta", "arch", "atlases",
                                               "scheme", "repeats", "folds", "seed", "c", "epochs")}
    sig_path = out / "signature.json"
    if sig_path.exists():
        previous = json.loads(sig_path.read_text())
        if previous != json.loads(json.dumps(signature)):
            raise SystemExit(f"{out} holds folds fitted with different settings; delete it or pick another --method")
    out.mkdir(parents=True, exist_ok=True)
    write_json(sig_path, signature)
    summary = run_cv(method, folds, data.labels, out, scheme=args.scheme)
    write_json(out / "method.json", {**vars(args), "raw": str(args.raw), "processed": str(args.processed),
                                     "runs": str(args.runs), "out": str(out),
                                     "chosen_c": getattr(method, "chosen_c", None)})
    if args.scheme == "loso":
        oof = np.load(out / "oof.npz")
        per_site = {}
        for site in np.unique(data.sites):
            sel = data.sites == site
            per_site[site] = {"n": int(sel.sum()),
                              "accuracy": float(((oof["prob"][0, sel] >= 0.5) == data.labels[sel]).mean())}
        write_json(out / "per_site.json", per_site)
    return summary
