import numpy as np
import pytest
import torch

from brainssl.smri.model import BilateralExchange, UNet3D, UNetConfig
from brainssl.trust.smri import (
    CONTRASTS,
    PERTURBATIONS,
    case_stats,
    exchange_stats,
    mirror_blind,
    perturb,
    record_exchange,
    saliency,
)


def _image(seed=0, size=(16, 16, 16)):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(4, *size, generator=g)
    brain = torch.zeros(1, *size, dtype=torch.bool)
    brain[:, 3:13, 3:13, 3:13] = True
    return image * brain


def test_perturbations_are_reproducible_and_keep_the_background_empty():
    image = _image()
    for name in PERTURBATIONS:
        a, b = perturb(image, name, 5), perturb(image, name, 5)
        assert torch.equal(a, b)
        assert a.shape == image.shape and torch.isfinite(a).all()
        assert (a[:, ~(image != 0).any(0)] == 0).all(), name           # nothing leaks outside the brain
    assert not torch.equal(perturb(image, "noise 0.5", 1), perturb(image, "noise 0.5", 2))


def test_missing_contrast_zeroes_exactly_that_channel():
    image = _image()
    for c, name in enumerate(CONTRASTS):
        out = perturb(image, f"no {name}", 0)
        assert (out[c] == 0).all()
        assert torch.equal(out[[i for i in range(4) if i != c]], image[[i for i in range(4) if i != c]])


def test_noise_level_is_what_it_says():
    image = torch.ones(4, 16, 16, 16)
    delta = perturb(image, "noise 0.5", 3) - image
    assert delta.std().item() == pytest.approx(0.5, abs=0.03)


def test_blur_keeps_a_constant_image_constant_inside():
    out = perturb(torch.ones(4, 16, 16, 16), "blur", 0)
    assert torch.allclose(out[:, 5:11, 5:11, 5:11], torch.ones(4, 6, 6, 6), atol=1e-5)


def _model():
    torch.manual_seed(0)
    model = UNet3D(UNetConfig(channels=(4, 8, 16), bilateral=True, bilateral_levels=(1, 2))).eval()
    for m in model.modules():                         # make the exchange blocks non-trivial
        if isinstance(m, BilateralExchange):
            torch.nn.init.normal_(m.mix[-1].weight, std=0.3)
    return model


def test_mirror_blind_changes_the_output_and_restores_the_model():
    model, image = _model(), _image()[None]
    with torch.no_grad():
        before = model(image)
        with mirror_blind():
            blind = model(image)
        after = model(image)
    assert torch.equal(before, after)
    assert not torch.allclose(before, blind)


def test_blind_exchange_equals_the_real_block_on_a_symmetric_feature_map():
    block = BilateralExchange(6)
    torch.nn.init.normal_(block.mix[-1].weight, std=0.3)
    x = torch.randn(1, 6, 8, 8, 8)
    sym = 0.5 * (x + x.flip(2))                       # a left-right symmetric feature map: its mirror is itself
    real_sym, real_x = block(sym), block(x)
    with mirror_blind():
        blind_sym, blind_x = block(sym), block(x)
    assert torch.allclose(real_sym, blind_sym, atol=1e-6)
    assert not torch.allclose(real_x, blind_x)        # on an asymmetric map the mirror matters


def test_record_exchange_sees_one_map_per_exchange_level():
    model, image = _model(), _image()[None]
    with torch.no_grad(), record_exchange() as store:
        model(image)
    assert len(store) == 2
    assert store[0]["delta"].shape == (8, 8, 8) and store[1]["delta"].shape == (4, 4, 4)


def test_exchange_stats_and_saliency_run_end_to_end():
    model, image = _model(), _image()
    regions = np.zeros((3, 16, 16, 16), np.uint8)
    regions[0, 5:10, 8:13, 5:10] = 1                  # an off-centre "tumour"
    levels = exchange_stats(model, image, regions, (16, 16, 16), amp=False)
    assert len(levels) == 2 and all("asymmetry_auc" in lv for lv in levels if lv)
    sal, share, box = saliency(model, image, regions, (16, 16, 16), amp=False)
    assert sal.shape == (16, 16, 16) and share.sum() == pytest.approx(1.0, abs=1e-4)


def test_case_stats_histogram_counts_and_volumes():
    prob = torch.zeros(3, 12, 12, 12)
    prob[:, 4:8, 4:8, 4:8] = 0.9
    first = prob.clone()
    regions = np.zeros((3, 12, 12, 12), np.uint8)
    regions[:, 4:8, 4:8, 4:8] = 1
    row, extra = case_stats(prob, first, regions)
    assert row["vol_pred_WT"] == row["vol_true_WT"] == 64
    assert row["unc_disagree_WT"] == pytest.approx(0.0, abs=1e-6)
    hist = extra["hist"]
    assert hist[0, 0].sum() > 64 and hist[0, 2].sum() == pytest.approx(64)       # the zone is wider than the tumour
