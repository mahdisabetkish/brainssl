"""Patch sampling for supervised training and for pretraining.

A patch spans the full 160-voxel left-right width of the crop and 128 x 128
voxels in the other two axes, so it always contains both hemispheres. Loading a
case from its compressed NIfTI file costs about a second of CPU time, and a
data loader with a handful of workers keeps up with the GPU without any cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from .preprocess import load_case, regions_from_labels


def _crop_patch(arrays: list[np.ndarray], centre: np.ndarray, size: tuple[int, int]) -> list[np.ndarray]:
    """Crop axes 1 and 2 (AP, SI) around ``centre``; axis 0 is kept whole."""
    out = []
    shape = arrays[0].shape[-3:]
    starts = []
    for axis, s in zip((1, 2), size):
        n = shape[axis]
        start = int(np.clip(centre[axis] - s // 2, 0, max(0, n - s)))
        starts.append(start)
    for a in arrays:
        sl = (..., slice(None), slice(starts[0], starts[0] + size[0]), slice(starts[1], starts[1] + size[1]))
        patch = a[sl]
        pad = [(0, 0)] * (patch.ndim - 2) + [(0, max(0, size[0] - patch.shape[-2])),
                                            (0, max(0, size[1] - patch.shape[-1]))]
        out.append(np.pad(patch, pad) if any(p[1] for p in pad[-2:]) else patch)
    return out


class _Base(Dataset):
    def __init__(self, index: pd.DataFrame, case_ids: list[str], samples: int,
                 patch: tuple[int, int] = (128, 128), seed: int = 0, lr_width: int = 160,
                 raw_root: str = "data/raw") -> None:
        rows = index.set_index("case_id")
        self.raw_root = raw_root
        self.rows = [rows.loc[c] for c in case_ids]
        self.samples = samples
        self.patch = patch
        self.lr_width = lr_width
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.samples

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            info = get_worker_info()
            wid = info.id if info is not None else 0
            # Different stream per worker and per epoch-restart of the worker pool.
            self._rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed() % 2**31)])
        return self._rng


class SegmentationPatches(_Base):
    """Random labelled patches; half are centred on a tumour voxel."""

    def __init__(self, *args, foreground: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.foreground = foreground

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self.rng
        row = self.rows[rng.integers(len(self.rows))]
        image, label, _ = load_case(row, lr_width=self.lr_width, raw_root=self.raw_root)
        assert label is not None
        if rng.random() < self.foreground and (label > 0).any():
            fg = np.argwhere(label > 0)
            centre = fg[rng.integers(len(fg))]
        else:
            brain = np.argwhere((image != 0).any(axis=0))
            centre = brain[rng.integers(len(brain))]
        image, label = _crop_patch([image, label], centre, self.patch)
        regions = regions_from_labels(label)
        return torch.from_numpy(image.astype(np.float16)), torch.from_numpy(regions)


class UnlabelledPatches(_Base):
    """Random patches centred inside the brain, for pretraining."""

    def __getitem__(self, i: int) -> torch.Tensor:
        rng = self.rng
        row = self.rows[rng.integers(len(self.rows))]
        image, _, _ = load_case(row, with_label=False, lr_width=self.lr_width, raw_root=self.raw_root)
        brain = np.argwhere((image != 0).any(axis=0))
        centre = brain[rng.integers(len(brain))]
        (image,) = _crop_patch([image], centre, self.patch)
        return torch.from_numpy(image.astype(np.float16))
