import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant, assert_auto_jitable, random_irreps


class SlowNorm(torch.nn.Module):
    r"""Slow norm using TensorProduct"""
    def __init__(self, irreps_in, epsilon=None, squared=False):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, 'uuu', False, ir.dim)
            for i, (mul, ir) in enumerate(irreps_in)
        ]

        self.tp = o3.TensorProduct(
            irreps_in,
            irreps_in,
            irreps_out,
            instr,
            normalization='component'
        )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.epsilon = None if epsilon is None else float(epsilon)
        self.squared = squared

    def forward(self, features):
        out = self.tp(features, features)
        if self.epsilon is not None:
            eps_squared = self.epsilon**2
            for_sqrt = out.clone()
            for_sqrt[out < eps_squared] = eps_squared
        else:
            for_sqrt = out
        if self.squared:
            return for_sqrt
        else:
            return for_sqrt.sqrt()


@pytest.mark.parametrize(
    "irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4)
)
@pytest.mark.parametrize("batchdim", [(4,), (1,), tuple(), (5, 3, 7)])
@pytest.mark.parametrize("eps, squared", [(None, True), (None, False), (1e-3, False)])
def test_norm_like_tp(irreps_in, batchdim, eps, squared):
    """Test that Norm gives the same results as the corresponding TensorProduct."""
    m = o3.Norm(irreps_in, epsilon=eps, squared=squared)
    m_true = SlowNorm(irreps_in, epsilon=eps, squared=squared)
    inp = torch.randn(batchdim + (m.irreps_in.dim,))
    out = m(inp)
    out_true = m_true(inp)
    assert out.shape == out_true.shape
    assert torch.allclose(
        out,
        out_true,
        atol={torch.float32: 1e-8, torch.float64: 1e-10}[torch.get_default_dtype()],
    )


def test_norm():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    m = o3.Norm(irreps_in)
    m(torch.randn(irreps_in.dim))
    assert_equivariant(m)
    assert_auto_jitable(m)


def test_epsilon():
    irreps_in = o3.Irreps("3x1o")
    inp = torch.zeros(irreps_in.dim)
    inp[4] = 1.0
    m = o3.Norm(irreps_in, epsilon=1e-6)
    norms = m(inp)
    assert torch.all(norms == torch.Tensor([1e-6, 1.0, 1e-6]))
