"""Run the connectome experiment matrix in assets/fmri_methods.csv.

Each row is one classifier under one protocol and becomes
runs/fmri/<method>_<scheme>. Finished folds are reused, so the script can be
started again after any interruption.

    python scripts/run_fmri.py                         # every row
    python scripts/run_fmri.py --only vicinal_ensemble tangent_ensemble
    python scripts/run_fmri.py --linear-only           # skip the GPU baselines
"""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
from pathlib import Path

from brainssl.fmri.run import main as cv_main

log = logging.getLogger("run_fmri")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matrix", type=Path, default=Path("assets/fmri_methods.csv"))
    parser.add_argument("--runs", type=Path, default=Path("runs/fmri"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--deep-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="appended to every call, e.g. -- --repeats 1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    extra = [a for a in args.extra if a != "--"]

    with open(args.matrix, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if args.only and row["method"] not in args.only:
            continue
        deep = "--model deep" in row["args"]
        if (args.linear_only and deep) or (args.deep_only and not deep):
            continue
        out = args.runs / f"{row['method']}_{row['scheme']}"
        if args.dry_run:
            print(f"{'done' if (out / 'summary.json').exists() else 'todo'}  {out.name}")
            continue
        argv = ["--method", row["method"], "--scheme", row["scheme"], "--runs", str(args.runs)]
        argv += shlex.split(row["args"]) + extra
        log.info("running %s", out.name)
        cv_main(argv)


if __name__ == "__main__":
    main()
