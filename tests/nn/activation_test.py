import pytest

import torch

from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.test import assert_equivariant, assert_auto_jitable, assert_normalized
from e3nn.util.jit import prepare


@pytest.mark.parametrize(
    "irreps_in,acts",
    [("256x0o", [torch.abs]), ("37x0e", [torch.tanh]), ("4x0e + 3x0o", [torch.nn.functional.silu, torch.abs])],
)
def test_activation(irreps_in, acts) -> None:
    irreps_in = o3.Irreps(irreps_in)

    def build_module(irreps_in, acts):
        return Activation(irreps_in, acts)

    a = build_module(irreps_in, acts)
    a_pt2 = prepare(build_module)(irreps_in, acts)
    assert_auto_jitable(a)
    assert_equivariant(a)

    inp = irreps_in.randn(13, -1)
    out = a(inp)
    out_pt2 = a_pt2(inp)
    for ir_slice, act in zip(irreps_in.slices(), acts):
        this_out = out[:, ir_slice]
        this_out2 = out_pt2[:, ir_slice]
        true_up_to_factor = act(inp[:, ir_slice])
        factors = this_out / true_up_to_factor
        factor_pt2 = this_out2 / true_up_to_factor
        assert torch.allclose(factors, factors[0])
        assert torch.allclose(factor_pt2, factors[0])

    assert_normalized(a)
