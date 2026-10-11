"""Fix the regularisation strength of the noise-aware model from development runs.

Rule, written down before the reported study is analysed: for each atlas, take
the C in the development grid with the highest mean AUC of the noise-aware model
(intercept refitted on the training scans) over 3 repeats of 10-fold CV with
split seeds 0-2; ties go to the smaller C, the stronger penalty. The reported
study uses split seeds 10-14, so its splits differ from these.

    python scripts/choose_fmri_c.py --dev runs/dev_fmri --out assets/fmri_choices.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from brainssl.fmri.abide import load_phenotypes

ATLASES = ("cc200", "cc400", "aal", "ho", "dosenbach160")
GRID = (0.0001, 0.0003)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dev", type=Path, default=Path("runs/dev_fmri"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("assets/fmri_choices.json"))
    args = parser.parse_args()
    y = load_phenotypes(args.raw / "abide")["label"].to_numpy()
    chosen, table = {}, {}
    for atlas in ATLASES:
        scores = {}
        for C in GRID:
            f = args.dev / f"{atlas}_vicinal_tta_recal_C{C}.npy"
            if f.exists():
                P = np.load(f)
                scores[C] = float(np.mean([roc_auc_score(y, p) for p in P]))
        if not scores:
            raise SystemExit(f"no development runs for {atlas}")
        best = max(scores.values())
        chosen[atlas] = min(C for C, s in scores.items() if s >= best - 1e-9)
        table[atlas] = {f"{C:g}": round(s, 4) for C, s in scores.items()}
    record = {"vicinal_C": chosen, "development_auc": table,
              "rule": "highest mean development AUC (split seeds 0-2), ties to the smaller C",
              "ensemble": "all five atlases, probabilities averaged", "primary_metric": "AUC"}
    args.out.write_text(json.dumps(record, indent=2) + "\n", newline="\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
