"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace


def tetris():
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = torch.tensor([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=torch.get_default_dtype())

    # apply random rotation
    pos = torch.einsum('zij,zaj->zai', o3.rand_matrix(len(pos)), pos)

    # put in torch_geometric format
    dataset = [Data(pos=pos) for pos in pos]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    return data, labels


def mean_std(name, x):
    print(f"{name} \t{x.mean():.1f} ± ({x.var(0).mean().sqrt():.1f}|{x.std():.1f})")


class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out) -> None:
        super().__init__()

        tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([3, 256, tp.weight_numel], torch.relu)
        self.tp = tp

    def forward(self, edge_src, edge_dst, node_features, edge_sh, edge_length_embedded, num_neighbors) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)
        edge_features = self.tp(node_features[edge_src], edge_sh, weight)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)
        return node_features


class Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)

        irreps = self.irreps_sh

        # First layer with gate
        gate = Gate(
            "16x0e + 16x0o", [torch.relu, torch.abs],  # scalar
            "8x0e + 8x0o + 8x0e + 8x0o", [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            "16x1o + 16x1e"  # non-scalars, num_irreps has to match with gates
        )
        self.conv = Convolution(irreps, self.irreps_sh, gate.irreps_in)
        self.gate = gate
        irreps = self.gate.irreps_out

        # Final layer
        self.final = Convolution(irreps, self.irreps_sh, "0o + 6x0e")

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 3.8  # typical number of neighbors
        num_nodes = 4  # typical number of nodes

        edge_src, edge_dst = radius_graph(data.pos, 2.1, data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalization='component', normalize=True)
        edge_length_embedded = soft_one_hot_linspace(edge_vec.norm(dim=1), 1.0, 2.0, 3) * 3**0.5

        x = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        x = self.conv(edge_src, edge_dst, x, edge_sh, edge_length_embedded, num_neighbors)
        x = self.gate(x)
        x = self.final(edge_src, edge_dst, x, edge_sh, edge_length_embedded, num_neighbors)

        return scatter(x, data.batch, dim=0).div(num_nodes**0.5)


def main():
    data, labels = tetris()
    f = Network()

    print(f)

    optim = torch.optim.Adam(f.parameters(), lr=1e-2)

    for step in range(200):
        pred = f(data)
        loss = (pred - labels).pow(2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            accuracy = pred.round().eq(labels).double().mean().item()
            print(f"{100 * accuracy:.1f}% accuracy")

    # Check equivariance
    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    print(f"Equivariance error = {error.abs().max().item():.1e}")


if __name__ == '__main__':
    main()


def test():
    torch.set_default_dtype(torch.float64)

    data, labels = tetris()
    f = Network()

    pred = f(data)
    loss = (pred - labels).pow(2).sum()
    loss.backward()

    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    assert error.abs().max() < 1e-10
