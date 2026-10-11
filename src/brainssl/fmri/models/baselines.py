"""Deep baselines for single-atlas connectome classification.

Each takes the connectivity matrices of one atlas and is tied to that atlas's
ROI count and ordering, which is exactly the restriction PACT removes.

* ``ConnectomeMLP``: two-layer perceptron on the vectorised upper triangle.
* ``BrainNetCNN``: Kawahara et al., NeuroImage 146 (2017). Edge-to-edge,
  edge-to-node and node-to-graph layers, with the layer widths of the paper.
* ``BrainNetworkTransformer``: Kan et al., NeurIPS 2022. Rows of the
  connectivity matrix are node features for a transformer encoder, read out by
  soft assignment to orthonormal cluster centres. The rows are first projected
  to a fixed width so that the number of heads need not divide the ROI count;
  apart from that the structure follows the paper.
"""

from __future__ import annotations

import torch
from torch import nn


class ConnectomeMLP(nn.Module):
    def __init__(self, n_rois: int, hidden: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("iu", torch.triu_indices(n_rois, n_rois, 1), persistent=False)
        d = self.iu.shape[1]
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, conn: torch.Tensor) -> torch.Tensor:
        return self.net(conn[:, self.iu[0], self.iu[1]])[:, 0]


class E2E(nn.Module):
    """Cross-shaped filter: out_ij = sum_k a_k A_ik + sum_k b_k A_kj, per channel pair."""

    def __init__(self, cin: int, cout: int, n: int) -> None:
        super().__init__()
        self.row = nn.Conv2d(cin, cout, (1, n))
        self.col = nn.Conv2d(cin, cout, (n, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.row(x) + self.col(x)          # (B, C, N, 1) + (B, C, 1, N) -> (B, C, N, N)


class BrainNetCNN(nn.Module):
    def __init__(self, n_rois: int, dropout: float = 0.5) -> None:
        super().__init__()
        act = nn.LeakyReLU(0.33)
        self.e2e1 = E2E(1, 32, n_rois)
        self.e2e2 = E2E(32, 64, n_rois)
        self.e2n = nn.Conv2d(64, 1, (1, n_rois))
        self.n2g = nn.Conv2d(1, 256, (n_rois, 1))
        self.head = nn.Sequential(act, nn.Dropout(dropout), nn.Linear(256, 128), act, nn.Dropout(dropout),
                                  nn.Linear(128, 30), act, nn.Linear(30, 1))
        self.act = act

    def forward(self, conn: torch.Tensor) -> torch.Tensor:
        x = conn[:, None]
        x = self.act(self.e2e1(x))
        x = self.act(self.e2e2(x))
        x = self.act(self.e2n(x))
        x = self.n2g(x).flatten(1)
        return self.head(x)[:, 0]


class OCRead(nn.Module):
    """Orthonormal cluster readout of node embeddings."""

    def __init__(self, dim: int, clusters: int) -> None:
        super().__init__()
        centres = torch.linalg.qr(torch.randn(dim, clusters))[0].T    # rows are orthonormal
        self.centres = nn.Parameter(centres)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        assign = torch.softmax(h @ self.centres.T, dim=-1)            # (B, N, K)
        return self.norm(assign.transpose(1, 2) @ h)                  # (B, K, D)


class BrainNetworkTransformer(nn.Module):
    def __init__(self, n_rois: int, dim: int = 128, heads: int = 4, layers: int = 2, clusters: int = 10,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_rois, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.readout = OCRead(dim, clusters)
        self.head = nn.Sequential(nn.Linear(clusters * dim, 256), nn.LeakyReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 32), nn.LeakyReLU(), nn.Linear(32, 1))

    def forward(self, conn: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(conn))
        return self.head(self.readout(h).flatten(1))[:, 0]
