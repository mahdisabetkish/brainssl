"""Residual 3D U-Net with an optional bilateral exchange at every scale.

Tensors are laid out as (N, C, X, Y, Z) in RAS orientation, so dimension 2 runs
left to right. Volumes are cropped symmetrically about the mid-sagittal plane
before they reach the network (see ``preprocess.py``), which makes a flip of
dimension 2 the reflection across that plane at every resolution level, as long
as the size along X is divisible by 2 ** (levels - 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

LR_DIM = 2


def _norm(channels: int) -> nn.Module:
    return nn.InstanceNorm3d(channels, affine=True)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = _norm(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), _norm(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class BilateralExchange(nn.Module):
    """Let every voxel read the features at its mirror image across the midline.

    For features f and their reflection m = R f, the pair (f, m) carries a
    representation of the two-element reflection group: R swaps the halves.
    f + m spans the invariant part and f - m the sign part; |f - m| is an
    invariant that measures how far a location departs from its contralateral
    twin. The block feeds (f, m, |f - m|) through a pointwise MLP and adds the
    result back to f. The last layer starts at zero, so an untrained block is the
    identity and a network with these blocks begins as the plain U-Net.
    """

    def __init__(self, channels: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or channels
        self.mix = nn.Sequential(
            nn.Conv3d(3 * channels, hidden, 1, bias=False),
            _norm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, channels, 1),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mirror = torch.flip(x, dims=(LR_DIM,))
        return x + self.mix(torch.cat([x, mirror, (x - mirror).abs()], dim=1))


@dataclass
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 3
    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    # Residual blocks per encoder level; None means one at full resolution and two below.
    blocks: tuple[int, ...] | None = None
    bilateral: bool = False
    # Encoder levels that get an exchange block. Level 0 is full resolution,
    # where the block costs most of a forward pass and adds little: asymmetry
    # worth reacting to spans more than a couple of voxels.
    bilateral_levels: tuple[int, ...] = (1, 2, 3, 4)
    deep_supervision: bool = True
    extra_heads: dict[str, int] = field(default_factory=dict)


class UNet3D(nn.Module):
    """Encoder-decoder with residual stages and transposed-convolution upsampling.

    ``forward`` returns the full-resolution logits, or with deep supervision in
    training mode a list ordered from full resolution downwards. ``extra_heads``
    adds named 1x1x1 output layers on the last decoder features, which the
    pretraining stage uses for reconstruction.
    """

    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        nb = cfg.blocks if cfg.blocks is not None else (1,) + (2,) * (len(ch) - 1)
        if len(nb) != len(ch):
            raise ValueError(f"{len(ch)} channel widths but {len(nb)} block counts")

        self.stem = nn.Sequential(
            nn.Conv3d(cfg.in_channels, ch[0], 3, padding=1, bias=False), _norm(ch[0]),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.encoder = nn.ModuleList()
        for i, (c, n) in enumerate(zip(ch, nb)):
            cin = ch[0] if i == 0 else ch[i - 1]
            stride = 1 if i == 0 else 2
            layers = [ResBlock(cin, c, stride)] + [ResBlock(c, c) for _ in range(n - 1)]
            self.encoder.append(nn.Sequential(*layers))
        self.exchange = nn.ModuleList(
            [BilateralExchange(c) if cfg.bilateral and i in cfg.bilateral_levels else nn.Identity()
             for i, c in enumerate(ch)]
        )

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(ch[i], ch[i - 1], 2, stride=2, bias=False))
            self.decoder.append(ResBlock(2 * ch[i - 1], ch[i - 1]))

        self.head = nn.Conv3d(ch[0], cfg.out_channels, 1)
        # Auxiliary heads on (up to) the two decoder levels below full resolution.
        self.aux_levels = [k for k in (1, 2) if k <= len(ch) - 2] if cfg.deep_supervision else []
        self.aux = nn.ModuleList([nn.Conv3d(ch[k], cfg.out_channels, 1) for k in self.aux_levels])
        self.extra = nn.ModuleDict({k: nn.Conv3d(ch[0], v, 1) for k, v in cfg.extra_heads.items()})

    @property
    def divisor(self) -> int:
        return 2 ** (len(self.cfg.channels) - 1)

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the network up to the last decoder layer.

        Returns the full-resolution features and the decoder outputs from the
        coarsest level upwards, which the deep supervision heads read.
        """
        x = self.stem(x)
        skips = []
        for stage, exchange in zip(self.encoder, self.exchange):
            x = exchange(stage(x))
            skips.append(x)
        decoded = []
        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            x = block(torch.cat([up(x), skip], dim=1))
            decoded.append(x)
        return x, decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        feats, decoded = self.features(x)
        logits = self.head(feats)
        if self.training and self.aux_levels:
            # decoded[-1 - k] sits at 1 / 2**k of the input resolution.
            return [logits] + [head(decoded[-1 - k]) for head, k in zip(self.aux, self.aux_levels)]
        return logits

    def forward_extra(self, x: torch.Tensor, name: str) -> torch.Tensor:
        feats, _ = self.features(x)
        return self.extra[name](feats)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
