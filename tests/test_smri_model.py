import torch

from brainssl.smri.model import BilateralExchange, UNet3D, UNetConfig


def test_exchange_starts_as_identity():
    block = BilateralExchange(8)
    x = torch.randn(2, 8, 10, 6, 6)
    assert torch.equal(block(x), x)


def test_exchange_sees_the_mirror_position():
    torch.manual_seed(0)
    block = BilateralExchange(4)
    torch.nn.init.normal_(block.mix[-1].weight)
    x = torch.randn(1, 4, 8, 4, 4)
    y0 = block(x)
    x[0, :, 1] += 1.0                   # perturb column 1 only
    y1 = block(x)
    changed = (y1 - y0).abs().sum(dim=(0, 1, 3, 4)) > 1e-6
    # Column 1 changes, and so does its mirror image 8 - 1 - 1 = 6. The instance
    # norm couples all positions weakly, so compare against the largest change.
    delta = (y1 - y0).abs().sum(dim=(0, 1, 3, 4))
    top = set(torch.topk(delta, 2).indices.tolist())
    assert top == {1, 6} and changed[1] and changed[6]


def test_unet_shapes_and_deep_supervision():
    cfg = UNetConfig(channels=(4, 8, 16, 32), blocks=(1, 1, 1, 1), bilateral=True, bilateral_levels=(1, 2, 3))
    model = UNet3D(cfg)
    x = torch.randn(1, 4, 16, 16, 16)
    outs = model.train()(x)
    assert [tuple(o.shape) for o in outs] == [(1, 3, 16, 16, 16), (1, 3, 8, 8, 8), (1, 3, 4, 4, 4)]
    assert model.eval()(x).shape == (1, 3, 16, 16, 16)
    shallow = UNet3D(UNetConfig(channels=(4, 8, 16), blocks=(1, 1, 1)))
    assert len(shallow.train()(x)) == 2


def test_bilateral_unet_is_plain_unet_at_init():
    torch.manual_seed(0)
    plain = UNet3D(UNetConfig(channels=(4, 8, 16), blocks=(1, 1, 1)))
    bilateral = UNet3D(UNetConfig(channels=(4, 8, 16), blocks=(1, 1, 1), bilateral=True,
                                  bilateral_levels=(1, 2)))
    state = {k: v for k, v in plain.state_dict().items()}
    bilateral.load_state_dict(state, strict=False)
    x = torch.randn(1, 4, 16, 16, 16)
    torch.testing.assert_close(plain.eval()(x), bilateral.eval()(x))
