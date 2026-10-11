"""ABIDE I phenotypes, quality control and ROI time series.

The time series are the Preprocessed Connectomes Project outputs for the C-PAC
pipeline with band-pass filtering and no global signal regression. Each subject
and atlas is a whitespace-separated text file with one column per ROI and a
header row of ROI labels such as ``#2001``. Reading 871 x 5 of those files takes
a few minutes, so ``build_cache`` packs each atlas into one ``.npz``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PHENOTYPE_FILE = "Phenotypic_V1_0b_preprocessed1.csv"
ATLASES = ("cc200", "cc400", "aal", "ho", "dosenbach160")

# Same rule as nilearn's fetch_abide_pcp(quality_checked=True); it keeps 871 subjects.
_QC_RULES = {
    "qc_rater_1": ("OK",),
    "qc_anat_rater_2": ("OK", "maybe"),
    "qc_func_rater_2": ("OK", "maybe"),
    "qc_anat_rater_3": ("OK",),
    "qc_func_rater_3": ("OK",),
}


def load_phenotypes(root: Path, quality_checked: bool = True) -> pd.DataFrame:
    """Subjects with a preprocessed scan, one row each, sorted by FILE_ID.

    Adds ``label`` (1 for autism, 0 for typically developing controls), ``site``
    (the 20 acquisition sites, with UCLA, UM and LEUVEN split into their
    sub-cohorts as in the release) and ``sex`` (0 male, 1 female).
    """
    df = pd.read_csv(root / PHENOTYPE_FILE)
    df = df[df["FILE_ID"] != "no_filename"].copy()
    if quality_checked:
        keep = np.ones(len(df), dtype=bool)
        for column, allowed in _QC_RULES.items():
            keep &= df[column].isin(allowed).to_numpy()
        df = df[keep]
    df["label"] = (df["DX_GROUP"] == 1).astype(np.int64)
    df["site"] = df["SITE_ID"].astype(str)
    df["sex"] = (df["SEX"] == 2).astype(np.int64)
    df["age"] = df["AGE_AT_SCAN"].astype(np.float64)
    columns = ["SUB_ID", "FILE_ID", "site", "label", "age", "sex", "func_mean_fd", "FIQ"]
    return df[columns].sort_values("FILE_ID").reset_index(drop=True)


def read_timeseries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One ``.1D`` file as a (time, roi) float32 array and its ROI labels."""
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").split()
    labels = np.array([int(h.lstrip("#")) for h in header], dtype=np.int64)
    data = np.loadtxt(path, skiprows=1, dtype=np.float32, ndmin=2)
    if data.shape[1] != len(labels):
        raise ValueError(f"{path}: {data.shape[1]} columns but {len(labels)} labels")
    return data, labels


@dataclass
class TimeSeriesSet:
    """All subjects' time series for one atlas, kept ragged along time."""

    atlas: str
    file_ids: np.ndarray       # (n,) str
    roi_labels: np.ndarray     # (r,) int
    data: np.ndarray           # (sum of T_i, r) float32
    offsets: np.ndarray        # (n + 1,) int64

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    @property
    def n_rois(self) -> int:
        return len(self.roi_labels)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def subset(self, file_ids: list[str] | np.ndarray) -> TimeSeriesSet:
        index = {f: i for i, f in enumerate(self.file_ids)}
        rows = [index[f] for f in file_ids]
        chunks = [self[i] for i in rows]
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
        return TimeSeriesSet(self.atlas, np.asarray(file_ids), self.roi_labels,
                             np.concatenate(chunks), offsets)


def _derivative_dir(root: Path, atlas: str, pipeline: str, strategy: str) -> Path:
    return root / pipeline / strategy / f"rois_{atlas}"


def build_cache(root: Path, out_dir: Path, atlases: tuple[str, ...] = ATLASES,
                pipeline: str = "cpac", strategy: str = "filt_noglobal") -> None:
    """Pack the text files of every subject with a scan into one npz per atlas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_ids = load_phenotypes(root, quality_checked=False)["FILE_ID"].tolist()
    for atlas in atlases:
        target = out_dir / f"{atlas}.npz"
        if target.exists():
            continue
        folder = _derivative_dir(root, atlas, pipeline, strategy)
        chunks, labels = [], None
        for file_id in file_ids:
            data, roi = read_timeseries(folder / f"{file_id}_rois_{atlas}.1D")
            if labels is None:
                labels = roi
            elif not np.array_equal(labels, roi):
                raise ValueError(f"{file_id}: ROI labels differ from the first subject for {atlas}")
            chunks.append(data)
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
        np.savez(target, file_ids=np.array(file_ids), roi_labels=labels,
                 data=np.concatenate(chunks), offsets=offsets)
        log.info("cached %s: %d subjects, %d ROIs", atlas, len(file_ids), len(labels))


def load_cache(cache_dir: Path, atlas: str) -> TimeSeriesSet:
    with np.load(cache_dir / f"{atlas}.npz") as z:
        return TimeSeriesSet(atlas, z["file_ids"], z["roi_labels"], z["data"], z["offsets"])
