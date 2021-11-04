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

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

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
        n_weight=100,
        n_input=10_000,
        atol=0.5
    )


def test_bias():
    irreps_in = o3.Irreps("2x0e + 1e + 2x0e + 0o")
    irreps_out = o3.Irreps("3x0e + 1e + 3x0e + 5x0e + 0o")
    m = o3.Linear(irreps_in, irreps_out, biases=[True, False, False, True, False])
    with torch.no_grad():
        m.bias[:].fill_(1.0)
    x = m(torch.zeros(irreps_in.dim))

    assert torch.allclose(x, torch.tensor([
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0
    ]))

    assert_equivariant(m)
    assert_auto_jitable(m)

    m = o3.Linear("0e + 0o + 1e + 1o", "10x0e + 0o + 1e + 1o", biases=True)

    assert_equivariant(m)
    assert_auto_jitable(m)
    assert_normalized(
        m,
        n_weight=100,
        n_input=10_000,
        atol=0.5,
        weights=[m.weight]
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


# We want to be sure to test a multiple-same L case, a single irrep case, and an empty irrep case
@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e"] + random_irreps(n=4)
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e"] + random_irreps(n=4)
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
        atol={torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()],
    )


def test_output_mask():
    irreps_in = o3.Irreps("1e + 2e")
    irreps_out = o3.Irreps("3e + 5x2o")
    m = o3.Linear(irreps_in, irreps_out)
    assert torch.all(m.output_mask == torch.zeros(m.irreps_out.dim, dtype=torch.bool))


def test_instructions_parameter():
    m = o3.Linear("4x0e + 3x4o", "1x2e + 4x0o")
    assert len(m.instructions) == 0
    assert not torch.any(m.output_mask)

    with pytest.raises(ValueError):
        m = o3.Linear(
            "4x0e + 3x4o",
            "1x2e + 4x0e",
            # invalid mixture of 0e and 2e
            instructions=[(0, 0)]
        )

    with pytest.raises(IndexError):
        m = o3.Linear(
            "4x0e + 3x4o",
            "1x2e + 4x0e",
            instructions=[(4, 0)]
        )


def test_empty_instructions():
    m = o3.Linear(
        o3.Irreps.spherical_harmonics(3),
        o3.Irreps.spherical_harmonics(3),
        instructions=[]
    )
    assert len(m.instructions) == 0
    assert not torch.any(m.output_mask)
    inp = m.irreps_in.randn(3, -1)
    out = m(inp)
    assert torch.all(out == 0.0)


def test_default_instructions():
    m = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
    )
    assert len(m.instructions) == 3
    assert torch.all(m.output_mask)
    ins_set = set((ins.i_in, ins.i_out) for ins in m.instructions)
    assert ins_set == {(0, 1), (1, 0), (2, 1)}
    assert set(ins.path_shape for ins in m.instructions) == {
        (4, 8), (2, 8), (3, 2)
    }


def test_instructions():
    m = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
        instructions=[(0, 1), (1, 0)]
    )
    inp = m.irreps_in.randn(3, -1)
    inp[:, :m.irreps_in[:2].dim] = 0.0
    out = m(inp)
    assert torch.allclose(out, torch.zeros(1))


def test_weight_view():
    m = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
        instructions=[(0, 1), (1, 0)]
    )
    inp = m.irreps_in.randn(3, -1)
    assert m.weight_view_for_instruction(0).shape == (4, 8)
    assert m.weight_view_for_instruction(1).shape == (3, 2)
    # Make weights going to output 0 all zeros
    with torch.no_grad():
        m.weight_view_for_instruction(1).fill_(0.0)
    out = m(inp)
    assert torch.allclose(out[:, :6], torch.zeros(1))

    for w in m.weight_views():
        with torch.no_grad():
            w.fill_(2.0)
    for i, ins, w in m.weight_views(yield_instruction=True):
        assert (w - 2.0).norm() == 0.0


def test_weight_view_unshared():
    m = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
        instructions=[(0, 1), (1, 0)],
        shared_weights=False
    )
    batchdim = 7
    inp = m.irreps_in.randn(batchdim, -1)
    weights = torch.randn(batchdim, m.weight_numel)
    assert m.weight_view_for_instruction(0, weights).shape == (batchdim, 4, 8)
    assert m.weight_view_for_instruction(1, weights).shape == (batchdim, 3, 2)
    # Make weights going to output 0 all zeros
    with torch.no_grad():
        m.weight_view_for_instruction(1, weights).fill_(0.0)
    out = m(inp, weights)
    assert torch.allclose(out[:, :6], torch.zeros(1))


def test_f():
    m = o3.Linear(
        "0e + 1e + 2e",
        "0e + 2x1e + 2e",
        f_in=44,
        f_out=25,
        _optimize_einsums=False
    )
    assert_equivariant(m, args_in=[torch.randn(10, 44, 9)])
    m = assert_auto_jitable(m)
    y = m(torch.randn(10, 44, 9))
    assert m.weight_numel == 4
    assert m.weight.numel() == 44 * 25 * 4
    assert 0.7 < y.pow(2).mean() < 1.4
