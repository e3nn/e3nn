from e3nn import o3
from e3nn.experimental import Linear
import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("irreps_in", ["0e + 1e"])
@pytest.mark.parametrize("irreps_out", ["2x0e + 2x1e"])
def test_linear(irreps_in, irreps_out):
    irreps_in = o3.Irreps(irreps_in)
    input = irreps_in.randn(-1)
    irreps_out = o3.Irreps(irreps_out)
    linear = o3.Linear(irreps_in, irreps_out)
    linear_compiled = torch.compile(Linear(irreps_in, irreps_out), fullgraph=True)
    torch.testing.assert_close(linear.weight_numel, linear_compiled.weight_numel)
    torch.testing.assert_close(linear(input), linear_compiled(input))
