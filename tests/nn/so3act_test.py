import pytest
import torch
from e3nn import o3
from e3nn.nn import SO3Activation
from e3nn.util.test import assert_equivariant
from e3nn.util.jit import compile, prepare


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
@pytest.mark.parametrize("act", [torch.tanh, lambda x: x**2])
def test_equivariance(act, lmax: int) -> None:
    m = SO3Activation(lmax, lmax, act, 6)

    assert_equivariant(m, ntrials=10, tolerance=0.04, irreps_in=so3_irreps(lmax), irreps_out=so3_irreps(lmax))


@pytest.mark.parametrize("aspect_ratio", [1, 2, 3, 4])
def test_identity(aspect_ratio) -> None:
    irreps = o3.Irreps([(2 * l + 1, (l, 1)) for l in range(5 + 1)])

    build_module = lambda: SO3Activation(5, 5, lambda x: x, 6, aspect_ratio=aspect_ratio)
    m = build_module()
    m = compile(m)

    m_pt2 = torch.compile(prepare(build_module)(), fullgraph=True)

    x = irreps.randn(-1)
    y = m(x)
    y2 = m_pt2(x)

    torch.allclose(y, y2)

    mse = (x - y).pow(2).mean()
    assert mse < 1e-5, mse
