import torch
from e3nn import o3, o2, experimental
import pytest


@pytest.mark.parametrize(
    "group, irreps_in1, irreps_in2",
    [
        (o3, "0e", "2x0e"),
        (o3, "0e", "2x0e + 3x1e"),
        (o3, "0e + 1e", "2x0e"),
        (o3, "0e + 1e", "2x0e + 3x1e"),
    ],
)
def test_fulltp(group, irreps_in1, irreps_in2):
    irreps_in1 = group.Irreps(irreps_in1)
    irreps_in2 = group.Irreps(irreps_in2)

    x = irreps_in1.randn(10, -1)
    y = irreps_in2.randn(10, -1)

    tp_pt2 = torch.compile(experimental.FullTensorProductv2(irreps_in1, irreps_in2), fullgraph=True)
    tp = group.FullTensorProduct(irreps_in1, irreps_in2)

    torch.testing.assert_close(tp_pt2(x, y), tp(x, y))
