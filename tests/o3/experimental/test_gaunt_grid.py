import torch
from e3nn import o3
import pytest
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize("irreps_in1", ["0e + 1o"])
@pytest.mark.parametrize("irreps_in2", ["0e + 1o"])
def test_gaunt_grid(irreps_in1, irreps_in2):
    x = o3.Irreps(irreps_in1).randn(10, -1)
    y = o3.Irreps(irreps_in2).randn(10, -1)

    gtp_pt2 = o3.experimental.GauntTensorProductS2Grid(irreps_in1, irreps_in2, res_beta=20, res_alpha=19)
    out = gtp_pt2(x, y)
    assert_equivariant(gtp_pt2, irreps_in=[irreps_in1, irreps_in2], irreps_out=gtp_pt2.irreps_out)
