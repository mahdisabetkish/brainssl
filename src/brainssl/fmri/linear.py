"""L2-regularised logistic regression solved on the GPU.

The tangent-space baselines fit thousands of these (folds x atlases x a grid of
C), with up to 77k features for the 392-region atlas. scikit-learn's lbfgs
solver takes seconds to tens of seconds per fit on the CPU; the same objective
minimised with torch's L-BFGS on the GPU takes well under a second. The
objective is scikit-learn's, so results match ``LogisticRegression(C=C)`` up to
solver tolerance (``tests/test_linear.py`` checks this):

    C * sum_i log(1 + exp(-s_i (w.x_i + b))) + 0.5 * ||w||^2
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GPULogisticRegression:
    def __init__(self, C: float = 1.0, max_iter: int = 200, tol: float = 1e-6,
                 device: str | torch.device = "cuda") -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
            ) -> GPULogisticRegression:
        if X.shape[0] < X.shape[1]:
            return self._fit_dual(X, y, sample_weight)
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        sw = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32,
                                                                device=self.device)
        w = torch.zeros(x.shape[1], device=self.device, requires_grad=True)
        b = torch.zeros(1, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = x @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, t, weight=sw, reduction="sum")
            # Divide through by C so the loss stays O(n) and gradients are well scaled.
            total = loss + 0.5 * (w @ w) / self.C
            total.backward()
            return total

        opt.step(closure)
        self.coef_ = w.detach().cpu().numpy()
        self.intercept_ = float(b.detach().cpu())
        return self

    def _fit_dual(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> GPULogisticRegression:
        """Same objective through the representer theorem, w = X^T a.

        With a few hundred subjects and tens of thousands of edges the problem
        lives in an n-dimensional subspace: the loss sees w only through the
        Gram matrix K = X X^T and ||w||^2 = a^T K a. Optimising over a costs
        O(n^2) per iteration instead of O(np), and the solution is the same.
        """
        x = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        t = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        sw = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float64,
                                                                device=self.device)
        gram = x @ x.T
        a = torch.zeros(x.shape[0], dtype=torch.float64, device=self.device, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ka = gram @ a
            loss = F.binary_cross_entropy_with_logits(ka + b, t, weight=sw, reduction="sum")
            total = loss + 0.5 * (a @ ka) / self.C
            total.backward()
            return total

        opt.step(closure)
        self.coef_ = (x.T @ a.detach()).float().cpu().numpy()
        self.intercept_ = float(b.detach().cpu())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        return np.asarray(X, dtype=np.float32) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.stack([1 - p, p], axis=1)
