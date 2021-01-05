import math

import torch
from e3nn import o3


def test_sh_equivariance1():
    r"""test
    - compose
    - spherical_harmonics_alpha_beta
    - irrep
    """
    torch.set_default_dtype(torch.float64)
    for l in range(7 + 1):
        a, b, _ = o3.rand_angles()
        alpha, beta, gamma = o3.rand_angles()

        ra, rb, _ = o3.compose_angles(alpha, beta, gamma, a, b, torch.tensor(0.0))
        Yrx = o3.spherical_harmonics_alpha_beta([l], ra, rb)

        Y = o3.spherical_harmonics_alpha_beta([l], a, b)
        DrY = o3.wigner_D(l, alpha, beta, gamma) @ Y

        assert (Yrx - DrY).abs().max() < 1e-10 * Y.abs().max()


def test_sh_is_in_irrep():
    torch.set_default_dtype(torch.float64)
    for l in range(4 + 1):
        a, b, _ = o3.rand_angles()
        Y = o3.spherical_harmonics_alpha_beta([l], a, b) * math.sqrt(4 * math.pi) / math.sqrt(2 * l + 1) * (-1) ** l
        D = o3.wigner_D(l, a, b, torch.zeros(()))
        assert (Y - D[:, l]).abs().max() < 1e-10
