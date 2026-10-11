"""Mask generators for masked image modelling on bilateral brain patches.

Masks are drawn on a coarse grid of ``block``-sized cubes and upsampled. Along
the left-right axis the grid is laid out symmetrically about the centre of the
patch, so the cube at index i and the cube at index n - 1 - i are mirror images
of each other across the midline.

``random`` masks each cube independently, which is the usual choice
(SimMIM-style masking for convolutional networks).

``contralateral`` masks are antisymmetric under the reflection R: for every
mirror pair of cubes exactly one is hidden, m(Rx) = 1 - m(x). Each hidden cube
therefore has a visible twin in the other hemisphere, and reconstructing it is
a question of how tissue on one side predicts tissue on the other. A further
fraction of pairs is hidden on both sides so the network cannot rely on the
mirror alone and still has to inpaint from local context.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _upsample(cells: torch.Tensor, shape: tuple[int, int, int], block: int) -> torch.Tensor:
    m = cells.repeat_interleave(block, 1).repeat_interleave(block, 2).repeat_interleave(block, 3)
    return m[:, : shape[0], : shape[1], : shape[2]]


def random_mask(batch: int, shape: tuple[int, int, int], ratio: float = 0.6, block: int = 16,
                device: torch.device | str = "cpu") -> torch.Tensor:
    grid = [-(-s // block) for s in shape]
    cells = (torch.rand(batch, *grid, device=device) < ratio).float()
    return _upsample(cells, shape, block)


def contralateral_mask(batch: int, shape: tuple[int, int, int], both: float = 0.2, block: int = 16,
                       device: torch.device | str = "cpu") -> torch.Tensor:
    """Expected masked fraction is 0.5 + both / 2 (0.6 with the default)."""
    nx = shape[0] // block
    if nx % 2 or shape[0] % block:
        raise ValueError("left-right size must be an even number of blocks")
    grid_yz = [-(-s // block) for s in shape[1:]]
    half = nx // 2
    left = (torch.rand(batch, half, *grid_yz, device=device) < 0.5).float()
    # Right half mirrors the complement of the left half, cube for cube.
    right = (1 - left).flip(1)
    cells = torch.cat([left, right], dim=1)
    pair_both = (torch.rand(batch, half, *grid_yz, device=device) < both).float()
    cells = torch.maximum(cells, torch.cat([pair_both, pair_both.flip(1)], dim=1))
    return _upsample(cells, shape, block)


def channel_dropout(batch: int, channels: int, p: float = 0.15,
                    device: torch.device | str = "cpu") -> torch.Tensor:
    """(B, C) mask of dropped contrasts; at least one contrast always stays."""
    drop = torch.rand(batch, channels, device=device) < p
    full = drop.all(dim=1)
    if full.any():
        keep = torch.randint(channels, (int(full.sum()),), device=device)
        drop[full.nonzero(as_tuple=True)[0], keep] = False
    return drop.float()


def apply_masks(image: torch.Tensor, spatial: torch.Tensor, dropped: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero hidden voxels; return the corrupted input and the loss weight map.

    The loss covers every voxel that was hidden spatially or through a dropped
    contrast, restricted to the brain.
    """
    hidden = torch.maximum(spatial[:, None], dropped[:, :, None, None, None])
    brain = (image != 0).any(dim=1, keepdim=True).float()
    corrupted = image * (1 - hidden)
    return corrupted, hidden * brain


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (F.mse_loss(pred.float(), target.float(), reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
