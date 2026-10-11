"""ROI coordinates for the ABIDE atlases.

The parcellation-agnostic model identifies a region by where it sits in MNI
space rather than by its index in one particular atlas, so every ROI needs a
centroid. Centroids are the centre of mass of each label in the atlas image the
Preprocessed Connectomes Project distributes, converted to millimetres with the
image affine. Volumes come along because they serve as quadrature weights when
a connectivity profile is integrated over space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from ..data.download import ATLAS_RESOURCES


@dataclass
class AtlasGeometry:
    name: str
    labels: np.ndarray        # (r,) int, in the column order of the time series
    centroids: np.ndarray     # (r, 3) float, MNI mm
    volumes: np.ndarray       # (r,) float, mm^3

    def hemisphere(self) -> np.ndarray:
        """-1 left, +1 right, 0 within 2 mm of the midline."""
        x = self.centroids[:, 0]
        return np.where(x < -2, -1, np.where(x > 2, 1, 0))


def atlas_geometry(atlas_dir: Path, name: str, labels: np.ndarray) -> AtlasGeometry:
    image_file, _ = ATLAS_RESOURCES[name]
    img = nib.load(atlas_dir / image_file)
    data = np.asarray(img.dataobj).astype(np.int64)
    if data.ndim == 4:
        data = data[..., 0]
    present = set(np.unique(data).tolist())
    missing = [int(l) for l in labels if int(l) not in present]
    if missing:
        raise ValueError(f"{name}: labels {missing[:5]} not found in {image_file}")
    index = [int(l) for l in labels]
    com = np.array(ndimage.center_of_mass(np.ones_like(data), data, index))
    counts = np.array(ndimage.sum(np.ones_like(data), data, index))
    voxel_volume = float(abs(np.linalg.det(img.affine[:3, :3])))
    centroids = nib.affines.apply_affine(img.affine, com)
    return AtlasGeometry(name, np.asarray(labels), centroids.astype(np.float32),
                         (counts * voxel_volume).astype(np.float32))
