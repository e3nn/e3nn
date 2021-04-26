import pytest

from typing import Optional

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant, assert_auto_jitable, random_irreps, assert_normalized


class SlowLinear(torch.nn.Module):
    r"""Inefficient implimentation of Linear relying on TensorProduct."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        internal_weights=None,
        shared_weights=None,
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()

        instr = [
            (i_in, 0, i_out, "uvw", True, 1.0)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        self.tp = o3.TensorProduct(
            irreps_in,
            "0e",
            irreps_out,
            instr,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
        )

        self.output_mask = self.tp.output_mask
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def forward(self, features, weight: Optional[torch.Tensor] = None):
        ones = torch.ones(
            features.shape[:-1] + (1,), dtype=features.dtype, device=features.device
        )
        return self.tp(features, ones, weight)


def test_linear():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = o3.Linear(irreps_in, irreps_out)
    m(torch.randn(irreps_in.dim))

    assert_equivariant(m)
    assert_auto_jitable(m)
    assert_normalized(
        m,
        n_weight=125,
        n_input=10_000,
        atol=0.3
    )


def test_single_out():
    l1 = o3.Linear("5x0e", "5x0e")
    l2 = o3.Linear("5x0e", "5x0e + 3x0o")
    with torch.no_grad():
        l1.weight[:] = l2.weight
    x = torch.randn(3, 5)
    out1 = l1(x)
    out2 = l2(x)
    assert out1.shape == (3, 5)
    assert out2.shape == (3, 8)
    assert torch.allclose(out1, out2[:, :5])
    assert torch.all(out2[:, 5:] == 0)


# We want to be sure to test a multiple-same L case and a single irrep case
@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4)
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e"] + random_irreps(n=4)
)
def test_linear_like_tp(irreps_in, irreps_out):
    """Test that Linear gives the same results as the corresponding TensorProduct."""
    m = o3.Linear(irreps_in, irreps_out)
    m_true = SlowLinear(irreps_in, irreps_out)
    with torch.no_grad():
        m_true.tp.weight[:] = m.weight
    inp = torch.randn(4, m.irreps_in.dim)
    assert torch.allclose(
        m(inp),
        m_true(inp),
        atol={torch.float32: 1e-7, torch.float64: 1e-10}[torch.get_default_dtype()],
    )


def test_output_mask():
    irreps_in = o3.Irreps("1e + 2e")
    irreps_out = o3.Irreps("3e + 5x2o")
    m = o3.Linear(irreps_in, irreps_out)
    assert torch.all(m.output_mask == torch.zeros(m.irreps_out.dim, dtype=torch.bool))
