import math

import torch

from e3nn import o3

from e3nn.util.test import assert_auto_jitable


def test_jit(float_tolerance) -> None:
    import e3nn
    sh = o3.SphericalHarmonicsAlphaBeta([0, 1, 2])

    a = torch.randn(5, 4)
    b = torch.randn(5, 4)

    torch._dynamo.reset()  # Clear cache from the previous run
    m_pt2 = torch.compile(sh, fullgraph=True)
    assert (sh(a, b) - m_pt2(a, b)).abs().max() < float_tolerance

    jited = assert_auto_jitable(sh)

    assert (sh(a, b) - jited(a, b)).abs().max() < float_tolerance


def test_sh_equivariance1(float_tolerance) -> None:
    r"""test
    - compose
    - spherical_harmonics_alpha_beta
    - irrep
    """
    for l in range(7 + 1):
        a, b, _ = o3.rand_angles()
        alpha, beta, gamma = o3.rand_angles()

        ra, rb, _ = o3.compose_angles(alpha, beta, gamma, a, b, torch.tensor(0.0))
        Yrx = o3.spherical_harmonics_alpha_beta(l, ra, rb)

        Y = o3.spherical_harmonics_alpha_beta(l, a, b)
        DrY = o3.wigner_D(l, alpha, beta, gamma) @ Y

        assert (Yrx - DrY).abs().max() < float_tolerance * Y.abs().max()


def test_sh_is_in_irrep(float_tolerance) -> None:
    for l in range(4 + 1):
        a, b, _ = o3.rand_angles()
        Y = o3.spherical_harmonics_alpha_beta(l, a, b) * math.sqrt(4 * math.pi) / math.sqrt(2 * l + 1)
        D = o3.wigner_D(l, a, b, torch.zeros(()))
        assert (Y - D[:, l]).abs().max() < float_tolerance


def test_sh_same(float_tolerance) -> None:
    for l in range(4 + 1):
        x = torch.randn(10, 3)
        a, b = o3.xyz_to_angles(x)

        y1 = o3.spherical_harmonics(l, x, True)
        y2 = o3.spherical_harmonics_alpha_beta(l, a, b)
        assert (y1 - y2).abs().max() < float_tolerance
