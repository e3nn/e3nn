import torch
from e3nn import o3
import pytest
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize("lmax", [1, 2])
def test_gaunt_grid(lmax):
    irreps_in = o3.Irreps.spherical_harmonics(lmax)
    x = irreps_in.randn(10, -1)
    y = irreps_in.randn(10, -1)

    gtp = o3.experimental.GauntTensorProductS2Grid(irreps_in, irreps_in, res_beta=20, res_alpha=19)
    gtp_pt2 = torch.compile(gtp, fullgraph=True, disable=True)
    out = gtp_pt2(x, y)
    assert_equivariant(gtp_pt2, irreps_in=[irreps_in, irreps_in], irreps_out=gtp_pt2.irreps_out)
