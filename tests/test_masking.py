import torch

from brainssl.smri.masking import apply_masks, channel_dropout, contralateral_mask, random_mask


def test_contralateral_mask_is_antisymmetric_without_double_masking():
    m = contralateral_mask(3, (160, 32, 32), both=0.0)
    torch.testing.assert_close(m + m.flip(1), torch.ones_like(m))


def test_contralateral_mask_ratio():
    m = contralateral_mask(64, (160, 128, 128), both=0.2)
    assert abs(m.mean().item() - 0.6) < 0.02
    # every voxel is hidden on at least one side of the midline
    assert ((m + m.flip(1)) >= 1).all()


def test_random_mask_ratio():
    m = random_mask(64, (160, 128, 128), ratio=0.6)
    assert abs(m.mean().item() - 0.6) < 0.02


def test_channel_dropout_keeps_one_contrast():
    d = channel_dropout(1000, 4, p=0.9)
    assert (d.sum(1) < 4).all()


def test_apply_masks_restricts_loss_to_brain():
    image = torch.zeros(1, 4, 16, 16, 16)
    image[:, :, 4:12, 4:12, 4:12] = 1.0
    spatial = torch.ones(1, 16, 16, 16)
    corrupted, weight = apply_masks(image, spatial, torch.zeros(1, 4))
    assert corrupted.abs().sum() == 0
    assert weight.sum() == 4 * 8 ** 3
