import pytest
import torch
from e3nn.o3 import Irreps

rotations = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, torch.pi / 2),
    (0.0, 0.0, torch.pi),
    (0.0, torch.pi / 2, 0.0),
    (0.0, torch.pi / 2, torch.pi / 2),
    (0.0, torch.pi / 2, torch.pi),
    (0.0, torch.pi, 0.0),
    (torch.pi / 2, 0.0, 0.0),
    (torch.pi / 2, 0.0, torch.pi / 2),
    (torch.pi / 2, 0.0, torch.pi),
    (torch.pi / 2, torch.pi / 2, 0.0),
]


def rotate_sparse_tensor(x, irreps, abc):
    """Perform a rotation of angles abc to a sparse tensor
    """
    from MinkowskiEngine import SparseTensor

    # rotate the coordinates (like vectors l=1)
    coordinates = x.C[:, 1:].to(x.F.dtype)
    coordinates = torch.einsum("ij,bj->bi", Irreps("1e").D_from_angles(*abc), coordinates)
    assert (coordinates - coordinates.round()).abs().max() < 1e-6
    coordinates = coordinates.round().to(torch.int32)
    coordinates = torch.cat([x.C[:, :1], coordinates], dim=1)

    # rotate the features (according to `irreps`)
    features = x.F
    features = torch.einsum("ij,bj->bi", irreps.D_from_angles(*abc), features)

    return SparseTensor(coordinates=coordinates, features=features)


@pytest.mark.parametrize("abc", rotations)
def test_equivariance(abc):
    pytest.importorskip("MinkowskiEngine")

    from MinkowskiEngine import SparseTensor
    from e3nn.nn.models.v2203.sparse_voxel_convolution import Convolution

    abc = torch.tensor(abc)

    irreps_in = Irreps("1e")
    irreps_out = Irreps("0e + 1e + 2e")

    conv = Convolution(
        irreps_in, irreps_out,
        irreps_sh="0e + 1e + 2e",
        diameter=7,
        num_radial_basis=3,
        steps=(1.0, 1.0, 1.0)
    )

    x1 = SparseTensor(
        coordinates=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.int32),
        features=irreps_in.randn(4, -1),
    )

    x2 = rotate_sparse_tensor(x1, irreps_in, abc)
    y2 = conv(x2)

    y1 = conv(x1)
    y1 = rotate_sparse_tensor(y1, irreps_out, abc)

    # check equivariance
    assert (y1.C - y2.C).abs().max() == 0
    assert (y1.F - y2.F).abs().max() < 1e-7 * y1.F.abs().max()
