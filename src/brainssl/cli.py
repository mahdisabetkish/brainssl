"""Command-line entry point: ``brainssl <command> ...``.

Every stage of the study is reachable from here, which is what the Nextflow
processes, the Docker entrypoint and the README call.

    brainssl download msd|abide        fetch the raw data
    brainssl prepare msd|abide         index, splits, caches, correlations, window connectomes
    brainssl smri pretrain|finetune    train the segmentation network
    brainssl smri evaluate             score a run on the test split
    brainssl fmri study                every linear connectome model on every atlas and split
    brainssl fmri cv                   score one connectome classifier under one protocol
    brainssl export ...                ONNX models and metadata for Rust and the browser
    brainssl report                    results tables and paired tests
    brainssl dashboard                 the data files behind the dashboard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _download(args: argparse.Namespace) -> None:
    from .data.download import fetch_abide, fetch_msd

    if args.dataset == "msd":
        fetch_msd(args.dest, workers=args.workers)
    else:
        fetch_abide(args.dest, workers=args.workers)


def _available_atlases(raw: Path) -> tuple[str, ...]:
    from .fmri.abide import ATLASES

    folder = raw / "abide" / "cpac" / "filt_noglobal"
    found = {p.name.removeprefix("rois_") for p in folder.glob("rois_*") if p.is_dir()}
    return tuple(a for a in ATLASES if a in found) or tuple(sorted(found))


def _prepare(args: argparse.Namespace) -> None:
    if args.dataset == "msd":
        from .smri.preprocess import build_index, make_splits, save_splits

        index = build_index(args.raw, args.out / "msd" / "index.csv", workers=args.workers)
        save_splits(make_splits(index, seed=args.seed), args.out / "msd" / "splits.json")
        return

    import numpy as np

    from .fmri.abide import build_cache, load_cache, load_phenotypes
    from .fmri.connectivity import batch_correlations, spd_log
    from .fmri.windows import cached_window_embeddings

    cache = args.out / "abide"
    atlases = _available_atlases(args.raw)
    build_cache(args.raw / "abide", cache, atlases)
    pheno = load_phenotypes(args.raw / "abide")
    for atlas in atlases:
        ts = load_cache(cache, atlas).subset(pheno["FILE_ID"].tolist())
        target = cache / f"corr_{atlas}.npy"
        if not target.exists():
            corr = batch_correlations([ts[i] for i in range(len(ts))])
            np.save(target, corr.astype(np.float32))
            np.save(cache / f"logs_{atlas}.npy", spd_log(corr).astype(np.float32))
        cached_window_embeddings(ts, cache, n_windows=args.windows, fraction=args.window_fraction)


def _smri(args: argparse.Namespace, rest: list[str]) -> None:
    if args.stage in ("pretrain", "finetune"):
        from .smri import train

        train.main([args.stage] + rest)
    else:
        from .smri.evaluate import main as evaluate_main

        evaluate_main(rest)


def _fmri(args: argparse.Namespace, rest: list[str]) -> None:
    if args.stage == "study":
        from .fmri.study import main as study_main

        study_main(rest)
    else:
        from .fmri.run import main as cv_main

        cv_main(rest)


def _export(args: argparse.Namespace, rest: list[str]) -> None:
    from .export.cli import main as export_main

    export_main(rest)


def _report(args: argparse.Namespace, rest: list[str]) -> None:
    from .report.cli import main as report_main

    report_main(rest)


def _dashboard(args: argparse.Namespace, rest: list[str]) -> None:
    from .report.dashboard import main as dashboard_main

    dashboard_main(rest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brainssl", description="brain MRI and fMRI study")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the raw datasets")
    p.add_argument("dataset", choices=["msd", "abide"])
    p.add_argument("--dest", type=Path, default=Path("data/raw"))
    p.add_argument("--workers", type=int, default=16)

    p = sub.add_parser("prepare", help="index, split and cache the data")
    p.add_argument("dataset", choices=["msd", "abide"])
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows", type=int, default=6, help="window connectomes per scan")
    p.add_argument("--window-fraction", type=float, default=0.7)

    sub.add_parser("smri", help="train or evaluate the segmentation network (see --help of each stage)",
                   add_help=False).add_argument("stage", choices=["pretrain", "finetune", "evaluate"])
    sub.add_parser("fmri", help="connectome classifiers (brainssl fmri study|cv --help)",
                   add_help=False).add_argument("stage", choices=["study", "cv"])
    sub.add_parser("export", help="export trained models (brainssl export --help)", add_help=False)
    sub.add_parser("report", help="tables and statistics (brainssl report --help)", add_help=False)
    sub.add_parser("dashboard", help="data files for the dashboard (brainssl dashboard --help)", add_help=False)

    args, rest = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    handlers = {"smri": _smri, "fmri": _fmri, "export": _export, "report": _report, "dashboard": _dashboard}
    if args.command in handlers:
        handlers[args.command](args, rest)
        return
    if rest:
        parser.error(f"unrecognised arguments: {' '.join(rest)}")
    {"download": _download, "prepare": _prepare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
