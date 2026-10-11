"""Connectome classifiers compared in the fMRI study.

Linear models
-------------
``LinearConnectome`` is L2-regularised logistic regression on a vectorised
connectivity matrix, in one of three embeddings:

* ``tangent``: affine-invariant tangent space at the training subjects'
  Frechet mean (Varoquaux et al., 2010), the reference pipeline of Abraham et
  al. (2017) and Dadi et al. (2019) and what nilearn implements;
* ``log``: log-Euclidean embedding vec(log C) (Arsigny et al., 2007);
* ``corr``: Fisher-transformed correlations.

With ``vicinal=True`` the same model is trained on each subject's temporal
resampling distribution: the full-scan connectome plus K window connectomes,
all carrying the subject's label, the windows sharing one unit of weight. This
is vicinal risk minimisation (Chapelle et al., 2001) with the vicinity of each
subject estimated from its own scan. For a linear score w.x the expected
squared perturbation under that vicinity is w' S_i w, with S_i the covariance of
the subject's window embeddings, so the training objective carries a data-driven
Tikhonov term that penalises exactly the directions in which a subject's
connectome is not reproducible within its own scan (cf. Bishop, 1995). With
``tta=True`` predictions are averaged over the full scan and its windows.

The regularisation strength is chosen by inner cross-validation on the training
fold, from the same grid and by the same rule for every linear model.

Deep models
-----------
``DeepConnectome`` trains one of the single-atlas networks in
``models/baselines.py`` on full-scan correlation matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .abide import load_cache, load_phenotypes
from .connectivity import TangentSpace, fisher_z, sym_to_vec
from .evaluate import Fold
from .windows import cached_window_embeddings

log = logging.getLogger(__name__)

C_GRID = (1e-4, 3e-4, 1e-3, 3e-3)


@dataclass
class ConnectomeData:
    """Everything the classifiers read, for the subjects in phenotype order."""

    labels: np.ndarray
    sites: np.ndarray
    atlases: tuple[str, ...]
    corr: dict[str, np.ndarray]          # (n, R, R) full-scan shrunk correlations
    logs: dict[str, np.ndarray]          # (n, R, R) their matrix logarithms
    windows: dict[str, np.ndarray] = field(default_factory=dict)   # (n, K, d) window log-embeddings

    @classmethod
    def load(cls, raw: Path, processed: Path, atlases: tuple[str, ...], windows: bool = True,
             n_windows: int = 6, fraction: float = 0.7) -> ConnectomeData:
        pheno = load_phenotypes(raw / "abide")
        cache = processed / "abide"
        corr, logs, wins = {}, {}, {}
        for atlas in atlases:
            corr[atlas] = np.load(cache / f"corr_{atlas}.npy").astype(np.float64)
            logs[atlas] = np.load(cache / f"logs_{atlas}.npy").astype(np.float64)
            if windows:
                ts = load_cache(cache, atlas).subset(pheno["FILE_ID"].tolist())
                wins[atlas] = cached_window_embeddings(ts, cache, n_windows, fraction)
        return cls(pheno["label"].to_numpy(), pheno["site"].to_numpy(), atlases, corr, logs, wins)

    @property
    def n(self) -> int:
        return len(self.labels)


def _dual_logreg(gram: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None, C: float,
                 iters: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """L2 logistic regression through the representer theorem (see linear.py)."""
    a = torch.zeros(gram.shape[0], dtype=gram.dtype, requires_grad=True)
    b = torch.zeros(1, dtype=gram.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=iters, tolerance_grad=1e-5, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ka = gram @ a
        loss = F.binary_cross_entropy_with_logits(ka + b, y, weight=weight, reduction="sum") + 0.5 * (a @ ka) / C
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach(), b.detach()


class LinearConnectome:
    DEFAULT_GRID = C_GRID

    def __init__(self, data: ConnectomeData, atlases: tuple[str, ...], embedding: str = "log",
                 vicinal: bool = False, tta: bool = False, c_grid: tuple[float, ...] = C_GRID,
                 inner_folds: int = 5, seed: int = 0, name: str | None = None) -> None:
        if vicinal or tta:
            if embedding != "log":
                raise ValueError("window views are log-Euclidean embeddings")
            if not data.windows:
                raise ValueError("load ConnectomeData with windows=True")
        self.data = data
        self.atlases = atlases
        self.embedding = embedding
        self.vicinal = vicinal
        self.tta = tta
        self.c_grid = c_grid
        self.inner_folds = inner_folds
        self.seed = seed
        self.name = name or f"{embedding}{'_vicinal' if vicinal else ''}{'_tta' if tta else ''}_{'+'.join(atlases)}"
        self.chosen_c: list[float] = []

    def _features(self, atlas: str, train: np.ndarray) -> np.ndarray:
        d = self.data
        if self.embedding == "tangent":
            ts = TangentSpace("frechet").fit(d.corr[atlas][train], logs=d.logs[atlas][train])
            return ts.transform(d.corr[atlas])
        if self.embedding == "log":
            return sym_to_vec(d.logs[atlas])
        if self.embedding == "corr":
            return fisher_z(d.corr[atlas])
        raise ValueError(self.embedding)

    def _atlas_probs(self, atlas: str, fold: Fold) -> np.ndarray:
        d = self.data
        tr, te = fold.train, fold.test
        full = self._features(atlas, tr)
        mu, sd = full[tr].mean(0), full[tr].std(0) + 1e-8
        k = d.windows[atlas].shape[1] if (self.vicinal or self.tta) else 0

        # Row blocks: full scans of all subjects, then every subject's windows.
        blocks = [(full - mu) / sd]
        if k:
            blocks.append(((d.windows[atlas] - mu) / sd).reshape(d.n * k, -1))
        z = torch.from_numpy(np.concatenate(blocks).astype(np.float32))
        win_rows = (lambda idx: (d.n + idx[:, None] * k + np.arange(k)[None]).ravel()) if k else None

        def train_rows(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not self.vicinal:
                return idx, d.labels[idx], np.ones(len(idx))
            rows = np.concatenate([idx, win_rows(idx)])
            y = np.concatenate([d.labels[idx], np.repeat(d.labels[idx], k)])
            w = np.concatenate([np.ones(len(idx)), np.full(len(idx) * k, 1.0 / k)])
            return rows, y, w

        def score_rows(idx: np.ndarray) -> list[np.ndarray]:
            views = [idx]
            if self.tta:
                views.append(win_rows(idx))
            return views

        needed = np.unique(np.concatenate([train_rows(tr)[0]] + score_rows(te)))
        pos = {r: i for i, r in enumerate(needed)}
        zn = z[torch.from_numpy(needed)]
        gram = (zn @ zn.T).double()
        cross = gram  # rows and columns both index `needed`

        def fit(idx: np.ndarray, C: float):
            rows, y, w = train_rows(idx)
            ix = torch.tensor([pos[r] for r in rows])
            a, b = _dual_logreg(gram[ix][:, ix], torch.from_numpy(y.astype(np.float64)),
                                torch.from_numpy(w.astype(np.float64)), C)
            return ix, a, b

        def predict(model, idx: np.ndarray) -> np.ndarray:
            ix, a, b = model
            out = []
            for rows in score_rows(idx):
                jx = torch.tensor([pos[r] for r in rows])
                logit = (cross[jx][:, ix] @ a + b).numpy()
                p = 1 / (1 + np.exp(-logit))
                out.append(p.reshape(len(idx), -1).mean(1))
            return np.mean(out, axis=0)

        # Inner CV for C, scored by AUC on full scans (plus windows with TTA).
        best_c, best_auc = self.c_grid[0], -np.inf
        if len(self.c_grid) > 1:
            strata = np.array([f"{s}|{l}" for s, l in zip(d.sites[tr], d.labels[tr])])
            inner = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed + fold.index)
            splits = list(inner.split(np.zeros(len(tr)), strata))
            for C in self.c_grid:
                p = np.zeros(len(tr))
                for itr, ite in splits:
                    p[ite] = predict(fit(tr[itr], C), tr[ite])
                auc = roc_auc_score(d.labels[tr], p)
                if auc > best_auc:
                    best_c, best_auc = C, auc
        self.chosen_c.append(best_c)
        return predict(fit(tr, best_c), te)

    def fit_predict(self, fold: Fold) -> dict[str, np.ndarray]:
        probs = np.stack([self._atlas_probs(a, fold) for a in self.atlases])
        return {"prob": probs.mean(0), "atlas_std": probs.std(0)}


class DeepConnectome:
    """A single-atlas deep baseline trained on full-scan correlation matrices."""

    def __init__(self, data: ConnectomeData, atlas: str, arch: str, epochs: int = 100, batch: int = 32,
                 lr: float = 1e-4, weight_decay: float = 1e-4, seed: int = 0,
                 device: str | None = None) -> None:
        from .models.baselines import BrainNetCNN, BrainNetworkTransformer, ConnectomeMLP

        self.data = data
        self.atlas = atlas
        self.arch = arch
        self.factory = {"mlp": ConnectomeMLP, "brainnetcnn": BrainNetCNN, "bnt": BrainNetworkTransformer}[arch]
        self.epochs, self.batch, self.lr, self.weight_decay, self.seed = epochs, batch, lr, weight_decay, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = f"{arch}_{atlas}"

    def fit_predict(self, fold: Fold) -> np.ndarray:
        torch.manual_seed(self.seed * 1000 + fold.repeat * 100 + fold.index)
        x = torch.from_numpy(self.data.corr[self.atlas].astype(np.float32))
        eye = torch.eye(x.shape[-1])
        x = x - eye                                   # the unit diagonal carries no information
        y = torch.from_numpy(self.data.labels.astype(np.float32))
        model = self.factory(x.shape[-1]).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        tr = torch.from_numpy(fold.train)
        gen = torch.Generator().manual_seed(int(fold.index))
        for _ in range(self.epochs):
            model.train()
            for chunk in tr[torch.randperm(len(tr), generator=gen)].split(self.batch):
                if len(chunk) < 2:
                    continue
                xb, yb = x[chunk].to(self.device), y[chunk].to(self.device)
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            te = torch.from_numpy(fold.test)
            return torch.sigmoid(model(x[te].to(self.device))).cpu().numpy()
