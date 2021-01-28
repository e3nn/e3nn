import torch
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from torch_geometric.data import Data, DataLoader


def test_gate_points_2101(float_tolerance):
    num_nodes = 5
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_attr = o3.Irreps("10x0e")

    dataset = [
        Data(
            pos=torch.randn(num_nodes, 3),
            x=irreps_in.randn(num_nodes, -1),
            z=irreps_attr.randn(num_nodes, -1)
        )
        for _ in range(10)
    ]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

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

    # Test equivariance with a full data pipeline:
    R = o3.rand_matrix()
    D_in = irreps_in.D_from_matrix(R)
    D_attr = irreps_attr.D_from_matrix(R)
    D_out = irreps_out.D_from_matrix(R)
    rotated_data = Data(pos=data.pos @ R.T, x=data.x @ D_in.T, z=data.z @ D_attr.T, batch=data.batch)

    pred = f(data)
    rotated_pred = f(rotated_data)

    assert (pred @ D_out.T - rotated_pred).abs().max() < float_tolerance
