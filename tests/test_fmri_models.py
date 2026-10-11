import pytest
import torch

from brainssl.fmri.models.baselines import BrainNetCNN, BrainNetworkTransformer, ConnectomeMLP


@pytest.mark.parametrize("cls", [ConnectomeMLP, BrainNetCNN, BrainNetworkTransformer])
def test_forward_and_backward(cls):
    torch.manual_seed(0)
    x = torch.randn(3, 12, 12)
    x = x + x.transpose(1, 2)
    model = cls(12)
    out = model(x)
    assert out.shape == (3,)
    out.sum().backward()
