"""Score a trained segmentation run on the held-out test split.

Uses the checkpoint with the best validation Dice, mirror test-time
augmentation, and both Dice and HD95 per region and per case. Per-case rows are
kept so that runs can be compared case by case with paired tests.

    python -m brainssl.smri.evaluate --run runs/smri/ft_ours_10pct

``--out`` writes the results to another directory (the pipeline keeps run
directories immutable); ``--raw``, ``--index`` and ``--splits`` override the
paths recorded in the run's config, for runs moved between machines.
``--save-cases`` also stores the probability maps of the listed scans, and
``--cases`` restricts scoring to a list of scans:

    python -m brainssl.smri.evaluate --run runs/smri/ft_ours_10pct --out runs/showcase/ft_ours_10pct \
        --cases BRATS_315 BRATS_478 --save-cases BRATS_315 BRATS_478
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils import write_json
from .model import UNet3D, UNetConfig
from .train import evaluate_cases

log = logging.getLogger(__name__)


def load_model(run: Path, checkpoint: str = "best.pt") -> UNet3D:
    cfg = json.loads((run / "config.json").read_text())
    model = UNet3D(UNetConfig(channels=tuple(cfg["channels"]), bilateral=cfg["bilateral"]))
    state = torch.load(run / checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def evaluate_run(run: Path, split: str = "test", checkpoint: str = "best.pt", mirror: bool = True,
                 save_cases: list[str] | None = None, out: Path | None = None, raw: str | None = None,
                 index_path: str | None = None, splits_path: str | None = None,
                 cases: list[str] | None = None) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    out = out or run
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(index_path or cfg["index"])
    splits = json.loads(Path(splits_path or cfg["splits"]).read_text())
    case_ids = list(cases) if cases else splits[split]
    model = load_model(run, checkpoint)
    window = (cfg.get("lr_width", 160),) + tuple(cfg.get("patch", (128, 128)))
    rows, predictions = evaluate_cases(model, index, case_ids, amp=cfg.get("amp", True), mirror=mirror,
                                       keep=set(save_cases or ()), window=window,
                                       raw_root=raw or cfg.get("raw", "data/raw"))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"eval_{split}.csv", index=False)
    metrics = [c for c in df.columns if c != "case_id"]
    summary = {"run": run.name, "split": split, "cases": len(df), "checkpoint": checkpoint, "mirror": mirror,
               "config": {k: cfg.get(k) for k in ("bilateral", "budget", "init", "iterations", "seed")},
               "mean": {m: float(df[m].mean()) for m in metrics},
               "median": {m: float(df[m].median()) for m in metrics}}
    write_json(out / f"eval_{split}.json", summary)
    if save_cases:
        pred_dir = out / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for case_id in save_cases:
            if case_id in predictions:
                prob, box = predictions[case_id]
                np.savez_compressed(pred_dir / f"{case_id}.npz", prob=prob,
                                    box=np.array([[s.start, s.stop] for s in box]))
    log.info("%s on %s: Dice WT %.3f TC %.3f ET %.3f | HD95 WT %.1f TC %.1f ET %.1f", run.name, split,
             summary["mean"]["dice_WT"], summary["mean"]["dice_TC"], summary["mean"]["dice_ET"],
             summary["mean"]["hd95_WT"], summary["mean"]["hd95_TC"], summary["mean"]["hd95_ET"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--save-cases", nargs="*", default=None,
                        help="store probability maps of these scans (ids, or a JSON file with a 'cases' list)")
    parser.add_argument("--cases", nargs="*", default=None, help="score only these scans of the split")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--splits", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    save = args.save_cases
    if save and len(save) == 1 and save[0].endswith(".json"):
        save = json.loads(Path(save[0]).read_text())["cases"]
    evaluate_run(args.run, args.split, args.checkpoint, mirror=not args.no_mirror, save_cases=save,
                 out=args.out, raw=args.raw, index_path=args.index, splits_path=args.splits, cases=args.cases)


if __name__ == "__main__":
    main()
