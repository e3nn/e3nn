"""Different paramters for the different atom types

based on `tetris_polynomial`

idea:
if we have num_z types of atoms we have num_z^2 types of edges.
Instead of having spherical harmonics for the edge attributes
we have num_z^2 times the spherical harmonics, all zero except for the type of the edge.

>>> test()
"""
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedTensorProduct, TensorProduct


class InvariantPolynomial(torch.nn.Module):
    def __init__(self, irreps_out, num_z, lmax) -> None:
        super().__init__()
        self.num_z = num_z
        self.lmax = lmax

        irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # to multiply the edge type one-hot with the spherical harmonics to get the edge attributes
        self.mul = TensorProduct(
            [(num_z**2, "0e")],
            irreps_sh,
            [(num_z**2, ir) for _, ir in irreps_sh],
            [
                (0, l, l, "uvu", False)
                for l in range(lmax + 1)
            ]
        )
        irreps_attr = self.mul.irreps_out

        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps(irreps_out)

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=irreps_sh,
            irreps_in2=irreps_attr,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=irreps_attr,
            irreps_out=irreps_out,
        )

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 3  # typical number of neighbors
        num_nodes = 4  # typical number of nodes
        num_z = self.num_z  # number of atom types

        # graph
        edge_src, edge_dst = radius_graph(data.pos, 10.0, data.batch)

        # spherical harmonics
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(list(range(self.lmax + 1)), edge_vec, normalization='component')

        # edge types
        edge_zz = num_z * data.z[edge_src] + data.z[edge_dst]  # from 0 to num_z^2 - 1
        edge_zz = torch.nn.functional.one_hot(edge_zz, num_z**2).mul(num_z)
        edge_zz = edge_zz.to(edge_sh.dtype)

        # edge attributes
        edge_attr = self.mul(edge_zz, edge_sh)

        # For each node, the initial features are the sum of the spherical harmonics of the neighbors
        node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(node_features[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp2(node_features[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each graph, all the node's features are summed
        return scatter(node_features, data.batch, dim=0).div(num_nodes**0.5)


def test():
    torch.set_default_dtype(torch.float64)

    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.5],
    ])

    # atom type
    z = torch.tensor([0, 1, 2, 2])

    dataset = [Data(pos=pos @ R.T, z=z) for R in o3.rand_matrix(10)]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    f = InvariantPolynomial("0e+0o", num_z=3, lmax=3)

    out = f(data)

    # expect invariant output
    assert out.std(0).max() < 1e-5
