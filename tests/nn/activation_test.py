import pytest
import functools

import torch
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.test import assert_equivariant, assert_auto_jitable, assert_normalized, assert_torch_compile

@pytest.mark.parametrize(
    "irreps_in,acts",
    [("256x0o", [torch.abs]), ("37x0e", [torch.tanh]), ("4x0e + 3x0o", [torch.nn.functional.silu, torch.abs])],
)
def test_activation(irreps_in, acts) -> None:
    irreps_in = o3.Irreps(irreps_in)

    a = Activation(irreps_in, acts)
    inp = irreps_in.randn(13, -1)

    assert_auto_jitable(a)

    assert_equivariant(a)

    out = a(inp)

    for ir_slice, act in zip(irreps_in.slices(), acts):
        this_out = out[:, ir_slice]
        true_up_to_factor = act(inp[:, ir_slice])
        factors = this_out / true_up_to_factor
        assert torch.allclose(factors, factors[0])

    assert_normalized(a)
