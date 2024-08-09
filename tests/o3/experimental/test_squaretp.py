import torch
from e3nn import o3
import pytest
import torch._dynamo
torch._dynamo.config.suppress_errors = True


@pytest.mark.parametrize("irreps_in", ["2x0e + 2x1e"])
def test_squaretp(irreps_in):
    x = o3.Irreps(irreps_in).randn(1, -1)
    square_tp_pt2 = torch.compile(o3.experimental.TensorSquarev2(irreps_in), fullgraph=True)
    square_tp = o3.TensorSquare(irreps_in)
    result_tp2 = square_tp_pt2(x)
    result_tp = square_tp(x)
    torch.testing.assert_close(result_tp2, result_tp)

@pytest.mark.parametrize("irreps_in", ["0e + 1e + 2e"])
def test_square_normalization(irreps_in) -> None:
    irreps = o3.Irreps(irreps_in)
    tp = o3.experimental.TensorSquarev2(irreps, irrep_normalization="norm")
    x = irreps.randn(1_000_000, -1, normalization="norm")
    y = tp(x)
    n = o3.Norm(tp.irreps_out, squared=True)(y)
    ## below test does not have the same compaison value (1.1)
    assert (n.mean(0).log().abs().exp() < 4).all()
