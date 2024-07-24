import torch
from e3nn import o3
import pytest


@pytest.mark.parametrize("irreps_in1, irreps_in2", [("10x0e", "5x0e + 5x1o"), ("2x0e + 1x1e", "2x0o + 1x1e")])
def test_elementwise_tp(irreps_in1, irreps_in2):

    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)

    x1 = irreps_in1.randn(5, -1)
    x2 = irreps_in2.randn(5, -1)

    tp = o3.ElementwiseTensorProduct(irreps_in1, irreps_in2)
    tp_pt2 = torch.compile(o3.experimental.ElementwiseTensorProductv2(irreps_in1, irreps_in2), fullgraph=True)
    result_tp = tp(x1, x2)
    result_tp2 = tp_pt2(x1, x2)

    assert tp.irreps_out == tp_pt2.irreps_out
    torch.testing.assert_close(result_tp, result_tp2)
