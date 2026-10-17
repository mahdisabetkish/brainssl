"""``brainssl trust``: calibration, subgroup, shortcut, interpretation and robustness audits.

    brainssl trust fmri       --study runs/fmri_study --out report/trust_fmri.json
    brainssl trust interpret  --connectome exports/connectome --out report/trust_interpret.json
    brainssl trust smri       --runs ours10=runs/smri/ft_ours_10pct scratch10=runs/smri/ft_unet_scratch_10pct \
                              --out report/trust_smri.json
    brainssl trust report     --dir report --out docs/trust_results.md

Each command reads trained models or out-of-fold predictions and writes one
JSON file; none of them changes a run directory. The analysis plan is in
docs/trust.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _runs(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--runs expects label=path, got {item!r}")
        out[label] = Path(path)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brainssl trust", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="part", required=True)

    p = sub.add_parser("fmri", help="calibration, selective prediction, subgroups and shortcuts")
    p.add_argument("--study", type=Path, default=Path("runs/fmri_study"))
    p.add_argument("--choices", type=Path, default=Path("assets/fmri_choices.json"))
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("report/trust_fmri.json"))

    p = sub.add_parser("interpret", help="weight maps, stability and faithfulness of the connectome classifier")
    p.add_argument("--connectome", type=Path, default=Path("exports/connectome"))
    p.add_argument("--choices", type=Path, default=Path("assets/fmri_choices.json"))
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--bootstraps", type=int, default=30)
    p.add_argument("--stability-atlases", nargs="*", default=["aal", "ho", "dosenbach160", "cc200"])
    p.add_argument("--deletion-atlases", nargs="*", default=["aal"])
    p.add_argument("--out", type=Path, default=Path("report/trust_interpret.json"))

    p = sub.add_parser("smri", help="uncertainty, calibration, failures, robustness and bilateral probes")
    p.add_argument("--runs", nargs="+", required=True, help="label=run_directory, one or more")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="proposed:reference labels whose robustness is compared case by case")
    p.add_argument("--robust-cases", type=int, default=24)
    p.add_argument("--layers-for", nargs="*", default=None,
                   help="scans whose uncertainty and saliency maps are stored (ids, or a JSON file with 'cases')")
    p.add_argument("--layers-root", type=Path, default=None)
    p.add_argument("--raw", default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--out", type=Path, default=Path("report/trust_smri.json"))

    p = sub.add_parser("report", help="write the results page from the audit files")
    p.add_argument("--dir", type=Path, default=Path("report"))
    p.add_argument("--out", type=Path, default=Path("docs/trust_results.md"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.part == "fmri":
        from .fmri import audit

        audit(args.study, args.choices, args.raw, args.out)
    elif args.part == "report":
        from .report import build

        build(args.dir, args.out)
    elif args.part == "interpret":
        from .interpret import interpret

        interpret(args.raw, args.processed, args.connectome, json.loads(args.choices.read_text()), args.out,
                  stability_atlases=tuple(args.stability_atlases), deletion_atlases=tuple(args.deletion_atlases),
                  n_boot=args.bootstraps)
    else:
        from .smri import audit

        layers = args.layers_for
        if layers and len(layers) == 1 and layers[0].endswith(".json"):
            layers = json.loads(Path(layers[0]).read_text())["cases"]
        audit(_runs(args.runs), args.out, layers_for=layers, layers_root=args.layers_root,
              robust_cases=args.robust_cases, raw=args.raw,
              index_path=args.index, splits_path=args.splits,
              pairs=[tuple(x.split(":")) for x in args.pairs])


if __name__ == "__main__":
    main()
