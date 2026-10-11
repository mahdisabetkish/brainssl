import numpy as np

from brainssl.smri.metrics import HD_PENALTY, dice, hd95


def test_dice_conventions():
    a = np.zeros((5, 5, 5), bool)
    assert dice(a, a) == 1.0
    b = a.copy()
    b[2, 2, 2] = True
    assert dice(b, a) == 0.0
    assert dice(b, b) == 1.0


def test_hd95_conventions_and_shift():
    a = np.zeros((40, 40, 40), bool)
    assert hd95(a, a) == 0.0
    b = a.copy()
    b[10:20, 10:20, 10:20] = True
    assert hd95(b, a) == HD_PENALTY
    c = np.roll(b, 3, axis=0)
    assert abs(hd95(b, c) - 3.0) < 1e-6
