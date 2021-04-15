import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant, assert_auto_jitable, random_irreps


class SlowNorm(torch.nn.Module):
    r"""Slow norm using TensorProduct"""
    def __init__(self, irreps_in):
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

    def forward(self, features):
        return self.tp(features, features).sqrt()


@pytest.mark.parametrize(
    "irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4)
)
@pytest.mark.parametrize("batchdim", [(4,), (1,), tuple(), (5, 3, 7)])
def test_norm_like_tp(irreps_in, batchdim):
    """Test that Norm gives the same results as the corresponding TensorProduct."""
    m = o3.Norm(irreps_in)
    m_true = SlowNorm(irreps_in)
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
