import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from brainssl.fmri.linear import GPULogisticRegression


@pytest.mark.parametrize("C", [0.01, 1.0])
def test_matches_sklearn(C, rng):
    x = rng.normal(size=(200, 30))
    y = (x[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    ours = GPULogisticRegression(C=C, tol=1e-9, max_iter=500).fit(x, y)
    ref = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit(x, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], atol=2e-3)
    np.testing.assert_allclose(ours.predict_proba(x)[:, 1], ref.predict_proba(x)[:, 1], atol=1e-3)
