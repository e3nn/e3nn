import torch
from e3nn import o3
import pytest


@pytest.mark.parametrize("irreps_in", ["0e", "0e + 1e"])
def test_squaretp(irreps_in):
    x = o3.Irreps(irreps_in).randn(1, -1)
    square_tp_pt2 = torch.compile(o3.experimental.TensorSquarev2(irreps_in), fullgraph=True)
    square_tp = o3.TensorSquare(irreps_in)
    result_tp2 = square_tp_pt2(x)
    result_tp = square_tp(x)
    torch.testing.assert_close(result_tp2, result_tp)
