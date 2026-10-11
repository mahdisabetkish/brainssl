import numpy as np
import pandas as pd

from brainssl.smri.preprocess import (
    _take,
    crop_box,
    estimate_midline,
    labels_from_regions,
    regions_from_labels,
    uncrop,
)


def _ellipsoid(centre_x: float, shape=(240, 240, 155)):
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((x - centre_x) / 65) ** 2 + ((y - 120) / 85) ** 2 + ((z - 75) / 60) ** 2 <= 1


def test_midline_recovers_known_centre():
    for c in (117.0, 119.5, 122.0):
        assert abs(estimate_midline(_ellipsoid(c)) - c) < 0.3


def test_crop_is_symmetric_about_midline():
    row = pd.Series({"midline": 119.5, "y0": 30, "y1": 200, "z0": 10, "z1": 140})
    box = crop_box(row)
    assert box[0].stop - box[0].start == 160
    assert (box[0].start + box[0].stop - 1) / 2 == 119.5
    assert all((s.stop - s.start) % 16 == 0 for s in box)


def test_crop_uncrop_round_trip():
    vol = np.random.default_rng(0).random((240, 240, 155)).astype(np.float32)
    box = (slice(40, 200), slice(20, 212), slice(-5, 155))
    crop = _take(vol, box)
    assert crop.shape == (160, 192, 160)
    assert (crop[:, :, :5] == 0).all()          # padded where the box leaves the volume
    back = uncrop(crop, box)
    inside = (slice(40, 200), slice(20, 212), slice(0, 155))
    np.testing.assert_array_equal(back[inside], vol[inside])
    assert back[:40].sum() == 0 and back[200:].sum() == 0


def test_region_label_round_trip():
    label = np.random.default_rng(0).integers(0, 4, size=(10, 10, 10)).astype(np.uint8)
    np.testing.assert_array_equal(labels_from_regions(regions_from_labels(label)), label)
