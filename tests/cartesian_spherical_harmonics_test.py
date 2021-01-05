import math
from functools import partial

import torch
from e3nn import o3


def test_weird_call():
    o3.spherical_harmonics([4, 1, 2, 3, 3, 1, 0], torch.randn(2, 1, 2, 3))


def test_zeros():
    assert torch.allclose(o3.spherical_harmonics([0, 1], torch.zeros(1, 3), normalization='norm'), torch.tensor([[1, 0, 0, 0.0]]))


def test_equivariance():
    torch.set_default_dtype(torch.float64)

    lmax = 5
    irreps = o3.Irreps.spherical_harmonics(lmax)
    x = torch.randn(2, 3)
    abc = o3.rand_angles()
    y1 = o3.spherical_harmonics(irreps, x @ o3.angles_to_matrix(*abc).T)
    y2 = o3.spherical_harmonics(irreps, x) @ irreps.D_from_angles(*abc).T

    assert (y1 - y2).abs().max() < 1e-10


def test_backwardable():
    torch.set_default_dtype(torch.float64)
    lmax = 3
    ls = list(range(lmax + 1))

    xyz = torch.tensor([
        [0., 0., 1.],
        [1.0, 0, 0],
        [0.0, 10.0, 0],
        [0.435, 0.7644, 0.023],
    ], requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(partial(o3.spherical_harmonics, ls), (xyz,), check_undefined_grad=False)


def test_normalization():
    torch.set_default_dtype(torch.float64)
    for l in range(10 + 1):
        n = o3.spherical_harmonics([l], torch.randn(3), 'integral', normalize=True).pow(2).mean()
        assert abs(n - 1 / (4 * math.pi)) < 1e-10

        n = o3.spherical_harmonics([l], torch.randn(3), 'norm', normalize=True).norm()
        assert abs(n - 1) < 1e-10

        n = o3.spherical_harmonics([l], torch.randn(3), 'component', normalize=True).pow(2).mean()
        assert abs(n - 1) < 1e-10


def test_closure():
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    torch.set_default_dtype(torch.float64)
    x = torch.randn(300_000, 3)
    Ys = [o3.spherical_harmonics(l, x, normalize=True) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * math.pi
            if l1 == l2:
                i = torch.eye(2 * l1 + 1)
                assert (m - i).abs().max() < 0.01
            else:
                assert m.abs().max() < 0.01


def test_parity():
    r"""
    (-1)^l Y(x) = Y(-x)
    """
    torch.set_default_dtype(torch.float64)
    for l in range(11 + 1):
        x = torch.randn(3)
        Y1 = (-1)**l * o3.spherical_harmonics(l, x)
        Y2 = o3.spherical_harmonics(l, -x)
        assert (Y1 - Y2).abs().max() < 1e-10
