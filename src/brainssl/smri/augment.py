"""Spatial and intensity augmentation applied to whole batches on the GPU.

Resampling a 4 x 160 x 128 x 128 patch with scipy takes over a second per
sample on the CPU, which would starve the GPU. Here a batch is transformed with
one ``affine_grid``/``grid_sample`` call.

The geometric transforms respect the bilateral layout the model relies on.
Flips along any axis keep the midline at the centre of the left-right axis.
Rotation about the left-right axis (pitch) moves the head in the sagittal plane
and leaves the mid-sagittal plane where it is; rotations about the other two
axes tilt that plane and are kept to a few degrees, which is about the residual
tilt one sees after rigid registration anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    p_flip: float = 0.5
    p_affine: float = 0.3
    pitch_deg: float = 15.0
    tilt_deg: float = 4.0
    scale: tuple[float, float] = (0.9, 1.1)
    p_gamma: float = 0.3
    gamma: tuple[float, float] = (0.7, 1.5)
    p_contrast: float = 0.3
    contrast: tuple[float, float] = (0.75, 1.25)
    p_noise: float = 0.15
    noise_std: float = 0.1
    p_blur: float = 0.1


def _rotation(axis: int, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    if axis == 0:
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == 1:
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    else:
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


class GPUAugment:
    def __init__(self, cfg: AugmentConfig | None = None) -> None:
        self.cfg = cfg or AugmentConfig()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, target: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """image (B, C, X, Y, Z) float; target (B, K, X, Y, Z) binary masks or None."""
        cfg = self.cfg
        b, device = image.shape[0], image.device

        for dim in (2, 3, 4):
            flip = torch.rand(b, device=device) < cfg.p_flip
            if flip.any():
                image[flip] = image[flip].flip(dim)
                if target is not None:
                    target[flip] = target[flip].flip(dim)

        do_affine = torch.rand(b, device=device) < cfg.p_affine
        if do_affine.any():
            n = int(do_affine.sum())
            # affine_grid works in (x, y, z) = (W, H, D) order, i.e. reversed tensor dims:
            # grid axis 0 is tensor dim 4 (Z), axis 2 is tensor dim 2 (left-right).
            pitch = _uniform(n, -cfg.pitch_deg, cfg.pitch_deg, device) * math.pi / 180
            tilt_a = _uniform(n, -cfg.tilt_deg, cfg.tilt_deg, device) * math.pi / 180
            tilt_b = _uniform(n, -cfg.tilt_deg, cfg.tilt_deg, device) * math.pi / 180
            rot = _rotation(2, pitch) @ _rotation(0, tilt_a) @ _rotation(1, tilt_b)
            scale = _uniform(n, *cfg.scale, device)
            theta = torch.zeros(n, 3, 4, device=device)
            theta[:, :, :3] = rot / scale[:, None, None]
            grid = F.affine_grid(theta, list(image[do_affine].shape), align_corners=False)
            image[do_affine] = F.grid_sample(image[do_affine], grid, mode="bilinear",
                                             padding_mode="zeros", align_corners=False)
            if target is not None:
                target[do_affine] = F.grid_sample(target[do_affine].float(), grid, mode="nearest",
                                                  padding_mode="zeros", align_corners=False).to(target.dtype)

        brain = (image != 0).any(dim=1, keepdim=True)
        c = image.shape[1]
        gamma_on = (torch.rand(b, c, device=device) < cfg.p_gamma)[..., None, None, None]
        if gamma_on.any():
            g = _uniform(b * c, *cfg.gamma, device).view(b, c, 1, 1, 1)
            lo = image.amin(dim=(2, 3, 4), keepdim=True)
            hi = image.amax(dim=(2, 3, 4), keepdim=True)
            unit = (image - lo) / (hi - lo + 1e-6)
            image = torch.where(gamma_on, unit.pow(g) * (hi - lo) + lo, image)
        contrast_on = (torch.rand(b, c, device=device) < cfg.p_contrast)[..., None, None, None]
        if contrast_on.any():
            k = _uniform(b * c, *cfg.contrast, device).view(b, c, 1, 1, 1)
            mean = image.mean(dim=(2, 3, 4), keepdim=True)
            image = torch.where(contrast_on, (image - mean) * k + mean, image)
        noise_on = (torch.rand(b, device=device) < cfg.p_noise).view(b, 1, 1, 1, 1)
        if noise_on.any():
            image = image + noise_on * cfg.noise_std * torch.randn_like(image)
        blur_on = torch.rand(b, device=device) < cfg.p_blur
        if blur_on.any():
            image[blur_on] = _gaussian_blur(image[blur_on], sigma=float(_uniform(1, 0.5, 1.0, device)))
        image = image * brain
        return image, target


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(2 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = len(k)
        weight = k.view(shape).expand(c, 1, *shape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = radius
        x = F.conv3d(F.pad(x, pad, mode="replicate"), weight, groups=c)
    return x
