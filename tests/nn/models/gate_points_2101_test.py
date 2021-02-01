import torch
from torch_geometric.data import Data

from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from e3nn.util.test import assert_equivariant


def test_gate_points_2101(float_tolerance):
    num_nodes = 5
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_attr = o3.Irreps("10x0e")
    irreps_out = o3.Irreps("2x0o + 2x1o + 2x2e")

    f = Network(
        irreps_in,
        o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o"),
        irreps_out,
        irreps_attr,
        o3.Irreps.spherical_harmonics(3),
        layers=3,
        max_radius=2.0,
        number_of_basis=5,
        radial_layers=2,
        radial_neurons=100,
        num_neighbors=4.0,
        num_nodes=num_nodes,
    )

    # Test equivariance:
    def wrapper(pos, x, z):
        data = Data(pos=pos, x=x, z=z, batch=torch.zeros(pos.shape[0], dtype=torch.long))
        return f(data)

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', irreps_in, irreps_attr],
        irreps_out=[irreps_out],
    )
