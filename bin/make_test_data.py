#!/usr/bin/env python3
"""Write a miniature copy of both datasets in their original on-disk formats.

The ``test`` profile of the pipeline and the CI jobs run every stage on this
in a few minutes on a CPU. It checks that the stages fit together, not the
science: volumes are 48 x 48 x 40 instead of 240 x 240 x 155, there are a dozen
subjects per site instead of dozens, and atlases have tens of regions.

    python bin/make_test_data.py --out data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _ellipsoid(shape, centre, radii):
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    return sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radii)) <= 1.0


def make_msd(out: Path, n_labelled: int, n_unlabelled: int, rng: np.random.Generator) -> None:
    task = out / "Task01_BrainTumour"
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    shape = (48, 48, 40)
    entries, tests = [], []
    for i in range(n_labelled + n_unlabelled):
        centre = ((shape[0] - 1) / 2 + rng.uniform(-0.5, 0.5), 24, 20)
        brain = _ellipsoid(shape, centre, (14, 18, 14))
        image = np.zeros(shape + (4,), dtype=np.float32)
        base = rng.normal(500, 40, size=shape + (4,)).astype(np.float32)
        image[brain] = base[brain]
        label = np.zeros(shape, dtype=np.uint8)
        side = rng.choice([-1, 1])
        tc = (centre[0] + side * rng.uniform(5, 8), rng.uniform(16, 32), rng.uniform(14, 26))
        oedema = _ellipsoid(shape, tc, (6, 6, 5)) & brain
        core = _ellipsoid(shape, tc, (3.5, 3.5, 3)) & brain
        enhancing = core & ~_ellipsoid(shape, tc, (2, 2, 1.5))
        label[oedema], label[core], label[enhancing] = 1, 2, 3
        image[oedema, 0] += 300      # FLAIR hyperintense oedema
        image[oedema, 3] += 250      # T2
        image[enhancing, 2] += 400   # T1Gd enhancing rim
        affine = np.eye(4)
        name = f"BRATS_{i + 1:03d}.nii.gz"
        if i < n_labelled:
            nib.save(nib.Nifti1Image(image, affine), task / "imagesTr" / name)
            nib.save(nib.Nifti1Image(label, affine), task / "labelsTr" / name)
            entries.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})
        else:
            nib.save(nib.Nifti1Image(image, affine), task / "imagesTs" / name)
            tests.append(f"./imagesTs/{name}")
    meta = {"name": "BRATS (synthetic)", "modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"},
            "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
            "numTraining": n_labelled, "numTest": n_unlabelled, "training": entries, "test": tests}
    (task / "dataset.json").write_text(json.dumps(meta, indent=1))


ATLAS_FILES = {"cc200": ("cc200_roi_atlas.nii.gz", "CC200_ROI_labels.csv"),
               "aal": ("aal_roi_atlas.nii.gz", "aal_labels.csv")}


def make_abide(out: Path, sites: int, per_site: int, rng: np.random.Generator) -> None:
    root = out / "abide"
    (root / "atlases").mkdir(parents=True, exist_ok=True)
    # Two coarse parcellations of a 10 x 12 x 10 grid at 12 mm in MNI-like space.
    shape = (10, 12, 10)
    affine = np.diag([12.0, 12.0, 12.0, 1.0])
    affine[:3, 3] = [-54, -72, -54]
    brain = _ellipsoid(shape, (4.5, 5.5, 4.5), (4.6, 5.6, 4.6))
    coords = np.argwhere(brain)
    labels_by_atlas = {}
    for atlas, n_rois, first in (("cc200", 16, 1), ("aal", 12, 2001)):
        centres = coords[rng.choice(len(coords), n_rois, replace=False)]
        nearest = ((coords[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        img = np.zeros(shape, dtype=np.int16)
        ids = np.arange(first, first + n_rois)
        img[tuple(coords.T)] = ids[nearest]
        image_file, label_file = ATLAS_FILES[atlas]
        nib.save(nib.Nifti1Image(img, affine), root / "atlases" / image_file)
        (root / "atlases" / label_file).write_text("\n".join(f"{i},roi{i}" for i in ids) + "\n")
        labels_by_atlas[atlas] = ids

    rows = []
    for s in range(sites):
        for k in range(per_site):
            asd = int(rng.random() < 0.5)
            file_id = f"SITE{s}_{50000 + s * 100 + k:07d}"
            rows.append({"SUB_ID": 50000 + s * 100 + k, "FILE_ID": file_id, "SITE_ID": f"SITE{s}",
                         "DX_GROUP": 1 if asd else 2, "AGE_AT_SCAN": float(rng.uniform(8, 30)),
                         "SEX": int(rng.choice([1, 2], p=[0.8, 0.2])), "func_mean_fd": float(rng.uniform(0.05, 0.3)),
                         "FIQ": float(rng.normal(105, 12)), "qc_rater_1": "OK", "qc_anat_rater_2": "OK",
                         "qc_func_rater_2": "OK", "qc_anat_rater_3": "OK", "qc_func_rater_3": "OK"})
            t = int(rng.integers(70, 110))
            factors = rng.normal(size=(t, 4))
            for atlas, ids in labels_by_atlas.items():
                n = len(ids)
                loading = rng.normal(size=(4, n)) * 0.6
                loading[0, : n // 3] += 0.5 * asd       # a weak group difference in one network
                series = factors @ loading + rng.normal(size=(t, n))
                folder = root / "cpac" / "filt_noglobal" / f"rois_{atlas}"
                folder.mkdir(parents=True, exist_ok=True)
                header = "\t".join(f"#{i}" for i in ids)
                np.savetxt(folder / f"{file_id}_rois_{atlas}.1D", series, fmt="%.6f", delimiter="\t",
                           header=header, comments="")
    import csv
    with open(root / "Phenotypic_V1_0b_preprocessed1.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--labelled", type=int, default=12)
    parser.add_argument("--unlabelled", type=int, default=4)
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--per-site", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    make_msd(args.out, args.labelled, args.unlabelled, rng)
    make_abide(args.out, args.sites, args.per_site, rng)
    print(f"synthetic data written to {args.out}")


if __name__ == "__main__":
    main()
