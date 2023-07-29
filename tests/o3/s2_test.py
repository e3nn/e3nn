import torch

import pytest
from e3nn.o3 import ToS2Grid, FromS2Grid, Irreps
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("lmax", [0, 1, 5, None])
def test_inverse1(float_tolerance, lmax, res_b, res_a) -> None:
    if lmax is None and res_b is None and res_a is None:
        return

    m = FromS2Grid((res_b, res_a), lmax)
    k = ToS2Grid(lmax, (res_b, res_a))

    res_b, res_a = m.res_beta, m.res_alpha
    x = torch.randn(res_b, res_a)
    x = k(m(x))  # remove high frequencies

    y = k(m(x))
    assert (x - y).abs().max().item() < float_tolerance


@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("lmax", [0, 1, 5, None])
def test_inverse2(float_tolerance, lmax, res_b, res_a) -> None:
    if lmax is None and res_b is None and res_a is None:
        return

    m = FromS2Grid((res_b, res_a), lmax)
    k = ToS2Grid(lmax, (res_b, res_a))
    lmax = m.lmax

    x = torch.randn((lmax + 1) ** 2)

    y = m(k(x))
    assert (x - y).abs().max().item() < float_tolerance


@pytest.mark.parametrize("res_a", [100, 101])
@pytest.mark.parametrize("res_b", [98, 100])
@pytest.mark.parametrize("lmax", [1, 5])
def test_equivariance(lmax, res_b, res_a) -> None:
    m = FromS2Grid((res_b, res_a), lmax)
    k = ToS2Grid(lmax, (res_b, res_a))

    def f(x):
        y = k(x)
        y = y.exp()
        return m(y)

    f.irreps_in = f.irreps_out = Irreps.spherical_harmonics(lmax)

    assert_equivariant(f)
