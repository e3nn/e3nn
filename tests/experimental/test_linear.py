from e3nn import o3
from e3nn.experimental import Linear
import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("irreps_in", ["1x0e + 1x2e"])
@pytest.mark.parametrize("irreps_out", ["2x0e + 3x2e"])
def test_linear(irreps_in, irreps_out):
    irreps_in = o3.Irreps(irreps_in)
    input = irreps_in.randn(-1)
    irreps_out = o3.Irreps(irreps_out)
    linear = o3.Linear(irreps_in, irreps_out)
    # linear.weight = torch.nn.Parameter(torch.ones_like(linear.weight)).to(device="cuda")
    linear_compiled = torch.compile(Linear(irreps_in, irreps_out), fullgraph=True, disable=True)
    # Not a fan of this so fix the abstraction
    # linear_compiled._weights[list(linear_compiled._weights.keys())[0]] = torch.nn.Parameter(
    #     torch.ones_like(list(linear_compiled._weights.values())[0])
    # ).to(device="cuda")
    torch.testing.assert_close(linear.weight_numel, linear_compiled.weight_numel)
    torch.testing.assert_close(linear(input), linear_compiled(input))
