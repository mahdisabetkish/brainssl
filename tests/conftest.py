import numpy as np
import pytest
import torch


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
