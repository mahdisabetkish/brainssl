"""Run the segmentation experiment matrix in assets/smri_runs.csv, in order.

Each row becomes runs/smri/<tag>; rows whose run already has a summary.json
are skipped, and an interrupted row resumes from its last checkpoint, so the
script can simply be started again after any interruption. Finished fine-tuning
runs are scored on the test split right away, keeping the probability maps of
the scans in assets/showcase_cases.json for the dashboard.

    python scripts/run_smri.py                     # everything
    python scripts/run_smri.py --only ft_ours_10pct ft_unet_scratch_10pct
    python scripts/run_smri.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from brainssl.smri.evaluate import evaluate_run
from brainssl.smri.train import FinetuneConfig, PretrainConfig, finetune, pretrain

log = logging.getLogger("run_smri")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matrix", type=Path, default=Path("assets/smri_runs.csv"))
    parser.add_argument("--runs", type=Path, default=Path("runs/smri"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    with open(args.matrix, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # probability maps of these scans are kept for the dashboard's viewer
    showcase = json.loads(Path("assets/showcase_cases.json").read_text())["cases"]
    for row in rows:
        tag = row["tag"]
        if args.only and tag not in args.only:
            continue
        out = args.runs / tag
        done = (out / "summary.json").exists()
        if args.dry_run:
            print(f"{'done' if done else 'todo'}  {tag}")
            continue
        common = dict(out_dir=str(out), bilateral=row["bilateral"] == "true", iterations=int(row["iterations"]),
                      seed=int(row["seed"]), workers=args.workers)
        if row["stage"] == "pretrain":
            pretrain(PretrainConfig(masking=row["masking"], **common))
        else:
            init = str(args.runs / row["init"] / "final.pt") if row["init"] else ""
            if init and not Path(init).exists():
                log.warning("skipping %s: pretraining %s has not finished", tag, row["init"])
                continue
            finetune(FinetuneConfig(budget=row["budget"], init=init, **common))
            if not (out / "eval_test.json").exists():
                evaluate_run(out, "test", save_cases=showcase)
        log.info("finished %s", tag)
        if (out / "eval_test.json").exists():
            summary = json.loads((out / "eval_test.json").read_text())["mean"]
            log.info("%s test Dice WT %.3f TC %.3f ET %.3f", tag, summary["dice_WT"], summary["dice_TC"],
                     summary["dice_ET"])


if __name__ == "__main__":
    main()
