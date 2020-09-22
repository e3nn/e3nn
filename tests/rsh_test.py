# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import math
from functools import partial

import torch
from e3nn import o3, rsh


def test_scipy_spherical_harmonics():
    with o3.torch_default_dtype(torch.float64):
        ls = [0, 1, 2, 3, 4, 5]
        beta = torch.linspace(1e-3, math.pi - 1e-3, 100, requires_grad=True).reshape(1, -1)
        alpha = torch.linspace(0, 2 * math.pi, 100, requires_grad=True).reshape(-1, 1)
        Y1 = rsh.spherical_harmonics_alpha_beta(ls, alpha, beta)
        Y2 = rsh.spherical_harmonics_alpha_beta(ls, alpha.detach(), beta.detach())
        assert (Y1 - Y2).abs().max() < 1e-10


def test_sh_is_in_irrep():
    with o3.torch_default_dtype(torch.float64):
        for l in range(4 + 1):
            a, b = 3.14 * torch.rand(2)  # works only for beta in [0, pi]
            Y = rsh.spherical_harmonics_alpha_beta([l], a, b) * math.sqrt(4 * math.pi) / math.sqrt(2 * l + 1) * (-1) ** l
            D = o3.irr_repr(l, a, b, 0)
            assert (Y - D[:, l]).norm() < 1e-10


def test_sh_cuda_single():
    if torch.cuda.is_available():
        with o3.torch_default_dtype(torch.float64):
            for l in range(10 + 1):
                x = torch.randn(10, 3)
                x_cuda = x.cuda()
                Y1 = rsh.spherical_harmonics_xyz([l], x)
                Y2 = rsh.spherical_harmonics_xyz([l], x_cuda).cpu()
                assert (Y1 - Y2).abs().max() < 1e-7
    else:
        print("Cuda is not available! test_sh_cuda_single skipped!")


def test_sh_cuda_ordered_full():
    if torch.cuda.is_available():
        with o3.torch_default_dtype(torch.float64):
            l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            x = torch.randn(10, 3)
            x_cuda = x.cuda()
            Y1 = rsh.spherical_harmonics_xyz(l, x)
            Y2 = rsh.spherical_harmonics_xyz(l, x_cuda).cpu()
            assert (Y1 - Y2).abs().max() < 1e-7
    else:
        print("Cuda is not available! test_sh_cuda_ordered_full skipped!")


def test_sh_cuda_ordered_partial():
    if torch.cuda.is_available():
        with o3.torch_default_dtype(torch.float64):
            l = [0, 2, 5, 7, 10]
            x = torch.randn(10, 3)
            x_cuda = x.cuda()
            Y1 = rsh.spherical_harmonics_xyz(l, x)
            Y2 = rsh.spherical_harmonics_xyz(l, x_cuda).cpu()
            assert (Y1 - Y2).abs().max() < 1e-7
    else:
        print("Cuda is not available! test_sh_cuda_ordered_partial skipped!")


def test_sh_parity():
    """
    (-1)^l Y(x) = Y(-x)
    """
    with o3.torch_default_dtype(torch.float64):
        for l in range(7 + 1):
            x = torch.randn(3)
            Y1 = (-1) ** l * rsh.spherical_harmonics_xyz([l], x)
            Y2 = rsh.spherical_harmonics_xyz([l], -x)
            assert (Y1 - Y2).abs().max() < 1e-10 * Y1.abs().max()


def test_sh_norm():
    with o3.torch_default_dtype(torch.float64):
        l_filter = list(range(15))
        Ys = [rsh.spherical_harmonics_xyz([l], torch.randn(10, 3)) for l in l_filter]
        s = torch.stack([Y.pow(2).mean(-1) for Y in Ys])
        d = s - 1 / (4 * math.pi)
        assert d.pow(2).mean().sqrt() < 1e-10


def test_sh_closure():
    """
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    with o3.torch_default_dtype(torch.float64):
        x = torch.randn(300000, 3)
        Ys = [rsh.spherical_harmonics_xyz([l], x) for l in range(0, 3 + 1)]
        for l1, Y1 in enumerate(Ys):
            for l2, Y2 in enumerate(Ys):
                m = (Y1.reshape(-1, 2 * l1 + 1, 1) * Y2.reshape(-1, 1, 2 * l2 + 1)).mean(0) * 4 * math.pi
                if l1 == l2:
                    i = torch.eye(2 * l1 + 1)
                    assert (m - i).abs().max() < 0.01
                else:
                    assert m.abs().max() < 0.01


def test_wigner_3j_sh_norm():
    with o3.torch_default_dtype(torch.float64):
        for l_out in range(3 + 1):
            for l_in in range(l_out, 4 + 1):
                for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                    Q = o3.wigner_3j(l_out, l_in, l_f)
                    Y = rsh.spherical_harmonics_xyz([l_f], torch.randn(3))
                    QY = math.sqrt(4 * math.pi) * Q @ Y
                    assert abs(QY.norm() - 1) < 1e-10


def test_sh_equivariance():
    """
    This test tests that
    - irr_repr
    - compose
    - spherical_harmonics
    are compatible

    Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
    with x = Z(a) Y(b) eta
    """
    for l in range(7):
        with o3.torch_default_dtype(torch.float64):
            a, b = torch.rand(2)
            alpha, beta, gamma = torch.rand(3)

            ra, rb, _ = o3.compose(alpha, beta, gamma, a, b, 0)
            Yrx = rsh.spherical_harmonics_alpha_beta([l], ra, rb)

            Y = rsh.spherical_harmonics_alpha_beta([l], a, b)
            DrY = o3.irr_repr(l, alpha, beta, gamma) @ Y

            assert (Yrx - DrY).abs().max() < 1e-10 * Y.abs().max()


def test_rsh_backwardable():
    lmax = 10
    Rs = [(1, l) for l in range(lmax + 1)]

    xyz = torch.tensor([
        [0., 0., 1.],
        [1.0, 0, 0],
        [0.0, 1.0, 0],
    ], requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(partial(rsh.spherical_harmonics_xyz, Rs), (xyz,), check_undefined_grad=False)
