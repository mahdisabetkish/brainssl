"""Region-based segmentation loss with deep supervision.

The network predicts three overlapping regions (whole tumour, tumour core,
enhancing tumour) with independent sigmoids rather than the four exclusive
labels with a softmax. The challenge metrics are defined on the regions, and
optimising them directly is the standard choice for BraTS since nnU-Net.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """1 - Dice per region, with sums taken over the batch and all voxels.

    Summing over the batch ("batch Dice") keeps the loss defined when a patch
    happens to contain no enhancing tumour.
    """
    prob = torch.sigmoid(logits.float())
    target = target.float()
    dims = (0, 2, 3, 4)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def region_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    return soft_dice_loss(logits, target) + bce


def deep_supervision_loss(outputs: list[torch.Tensor] | torch.Tensor, target: torch.Tensor,
                          weights: tuple[float, ...] = (1.0, 0.5, 0.25)) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return region_loss(outputs, target)
    total, norm = 0.0, 0.0
    for out, w in zip(outputs, weights):
        t = target if out.shape[2:] == target.shape[2:] else \
            F.interpolate(target.float(), size=out.shape[2:], mode="nearest")
        total = total + w * region_loss(out, t)
        norm += w
    return total / norm
