import math
import io

import functools

import pytest
import torch

from e3nn import o3
from e3nn import set_optimization_defaults, get_optimization_defaults
from e3nn.util.test import assert_auto_jitable, assert_equivariant, assert_torch_compile


def test_weird_call() -> None:
    o3.spherical_harmonics([4, 1, 2, 3, 3, 1, 0], torch.randn(2, 1, 2, 3), False)


def test_weird_irreps() -> None:
    # string input
    o3.spherical_harmonics("0e + 1o", torch.randn(1, 3), False)

    # Weird multipliciteis
    irreps = o3.Irreps("1x0e + 4x1o + 3x2e")
    out = o3.spherical_harmonics(irreps, torch.randn(7, 3), True)
    assert out.shape[-1] == irreps.dim

    # Bad parity
    with pytest.raises(ValueError):
        # L = 1 shouldn't be even for a vector input
        o3.SphericalHarmonics(
            irreps_out="1x0e + 4x1e + 3x2e",
            normalize=True,
            normalization="integral",
            irreps_in="1o",
        )

    # Good parity but psuedovector input
    _ = o3.SphericalHarmonics(irreps_in="1e", irreps_out="1x0e + 4x1e + 3x2e", normalize=True)

    # Invalid input
    with pytest.raises(ValueError):
        _ = o3.SphericalHarmonics(irreps_in="1e + 3o", irreps_out="1x0e + 4x1e + 3x2e", normalize=True)  # invalid


def test_zeros() -> None:
    assert torch.allclose(
        o3.spherical_harmonics([0, 1], torch.zeros(1, 3), False, normalization="norm"), torch.tensor([[1, 0, 0, 0.0]])
    )


def test_equivariance(float_tolerance) -> None:
    lmax = 5
    irreps = o3.Irreps.spherical_harmonics(lmax)
    x = torch.randn(2, 3)
    abc = o3.rand_angles()
    y1 = o3.spherical_harmonics(irreps, x @ o3.angles_to_matrix(*abc).T, False)
    y2 = o3.spherical_harmonics(irreps, x, False) @ irreps.D_from_angles(*abc).T

    assert (y1 - y2).abs().max() < 10 * float_tolerance


def test_backwardable() -> None:
    lmax = 3
    ls = list(range(lmax + 1))

    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0, 0],
            [0.0, 10.0, 0],
            [0.435, 0.7644, 0.023],
        ],
        requires_grad=True,
        dtype=torch.float64,
    )

    def func(pos):
        return o3.spherical_harmonics(ls, pos, False)

    assert torch.autograd.gradcheck(func, (xyz,), check_undefined_grad=False)


@pytest.mark.parametrize("l", range(10 + 1))
def test_normalization(float_tolerance, l) -> None:
    n = o3.spherical_harmonics(l, torch.randn(3), normalize=True, normalization="integral").pow(2).mean()
    assert abs(n - 1 / (4 * math.pi)) < float_tolerance

    n = o3.spherical_harmonics(l, torch.randn(3), normalize=True, normalization="norm").norm()
    assert abs(n - 1) < float_tolerance

    n = o3.spherical_harmonics(l, torch.randn(3), normalize=True, normalization="component").pow(2).mean()
    assert abs(n - 1) < float_tolerance


def test_closure() -> None:
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    x = torch.randn(1_000_000, 3)
    Ys = [o3.spherical_harmonics(l, x, True) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * math.pi
            if l1 == l2:
                i = torch.eye(2 * l1 + 1)
                assert (m - i).abs().max() < 0.01
            else:
                assert m.abs().max() < 0.01


@pytest.mark.parametrize("l", range(11 + 1))
def test_parity(float_tolerance, l) -> None:
    r"""
    (-1)^l Y(x) = Y(-x)
    """
    x = torch.randn(3)
    Y1 = (-1) ** l * o3.spherical_harmonics(l, x, False)
    Y2 = o3.spherical_harmonics(l, -x, False)
    assert (Y1 - Y2).abs().max() < float_tolerance


@pytest.mark.parametrize("l", range(9 + 1))
def test_recurrence_relation(float_tolerance, l) -> None:
    if torch.get_default_dtype() != torch.float64 and l > 6:
        pytest.xfail("we expect this to fail for high l and single precision")

    x = torch.randn(3, requires_grad=True)

    a = o3.spherical_harmonics(l + 1, x, False)

    b = torch.einsum("ijk,j,k->i", o3.wigner_3j(l + 1, l, 1), o3.spherical_harmonics(l, x, False), x)

    alpha = b.norm() / a.norm()

    assert (a / a.norm() - b / b.norm()).abs().max() < 10 * float_tolerance

    def f(x):
        return o3.spherical_harmonics(l + 1, x, False)

    a = torch.autograd.functional.jacobian(f, x)

    b = (l + 1) / alpha * torch.einsum("ijk,j->ik", o3.wigner_3j(l + 1, l, 1), o3.spherical_harmonics(l, x, False))

    assert (a - b).abs().max() < 100 * float_tolerance


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [True, False])
def test_module(normalization, normalize) -> None:
    l = o3.Irreps("0e + 1o + 3o")

    sp = o3.SphericalHarmonics(l, normalize, normalization)
    xyz = torch.randn(11, 3)

    sp_jit = assert_auto_jitable(sp)
    assert torch.allclose(sp_jit(xyz), o3.spherical_harmonics(l, xyz, normalize, normalization))
    assert_equivariant(sp)

    sp_pt2 = assert_torch_compile("inductor", functools.partial(o3.SphericalHarmonics, l, normalize, normalization), xyz)

    assert torch.allclose(sp_pt2(xyz), o3.spherical_harmonics(l, xyz, normalize, normalization))


@pytest.mark.parametrize("jit_mode", ["inductor", "eager"])
def test_pickle(jit_mode):
    l = o3.Irreps("0e + 1o + 3o")
    # Turning off the torch.jit.script in CodeGenMix to enable torch.compile.
    jit_mode_before = get_optimization_defaults()["jit_mode"]
    try:
        # Cannot pickle with compiled submodules
        set_optimization_defaults(jit_mode=jit_mode)
        sp = o3.SphericalHarmonics(l, normalization="integral", normalize=True)
        buffer = io.BytesIO()
        torch.save(sp, buffer)
    finally:
        set_optimization_defaults(jit_mode=jit_mode_before)
