"""(Ready to use) model with self-interactions and gates

Exact equivariance to :math:`E(3)`

>>> test()
"""
import math
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate, Linear, TensorProduct
from e3nn.math import gaussian_basis_projection


class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_edge, irreps_out, number_of_basis, radial_layers, radial_neurons, num_neighbors) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.si = Linear(self.irreps_in, self.irreps_out)

        self.lin1 = Linear(self.irreps_in, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge,
            irreps_mid,
            instructions,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], torch.relu, 1 / number_of_basis)
        self.tp = tp

        self.lin2 = Linear(irreps_mid, irreps_out)

    def forward(self, x, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        si = self.si(x)
        x = self.lin1(x)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x)).div(self.num_neighbors**0.5)

        x = self.lin2(x)
        return si + 0.5 * x


def smooth_transition(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class Network(torch.nn.Module):
    def __init__(
            self,
            irreps_in,
            irreps_out,
            irreps_hidden,
            irreps_sh,
            layers,
            max_radius,
            number_of_basis,
            radial_layers,
            radial_neurons,
            num_neighbors,
            num_nodes
        ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)

        irreps = self.irreps_in

        act = {
            1: torch.relu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.relu,
            -1: torch.abs,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_sh, ir)])
            irreps_nonscalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0])
            ir = "0e" if tp_path_exists(irreps, self.irreps_sh, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_nonscalars])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_nonscalars  # non-scalars
            )
            conv = Convolution(irreps, self.irreps_sh, gate.irreps_in, number_of_basis, radial_layers, radial_neurons, num_neighbors)
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(Convolution(irreps, self.irreps_sh, self.irreps_out, number_of_basis, radial_layers, radial_neurons, num_neighbors))

    def forward(self, data) -> torch.Tensor:
        edge_src, edge_dst = radius_graph(data.pos, self.max_radius, data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalization='component', normalize=True)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = gaussian_basis_projection(edge_length, 0.0, self.max_radius, self.number_of_basis)
        edge_attr = smooth_transition(edge_length / self.max_radius)[:, None] * edge_sh

        x = data.x
        for lay in self.layers:
            x = lay(x, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return scatter(x, data.batch, dim=0).div(self.num_nodes**0.5)


def test():
    torch.set_default_dtype(torch.float64)

    num_nodes = 5
    irreps_in = o3.Irreps("3x0e + 2x1o")

    dataset = [Data(pos=torch.randn(num_nodes, 3), x=torch.randn(num_nodes, irreps_in.dim)) for _ in range(10)]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    irreps_out = o3.Irreps("2x0o + 2x1o + 2x2e")

    f = Network(
        irreps_in,
        irreps_out,
        o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o"),
        o3.Irreps.spherical_harmonics(3),
        layers=3,
        max_radius=2.0,
        number_of_basis=5,
        radial_layers=2,
        radial_neurons=100,
        num_neighbors=4.0,
        num_nodes=num_nodes,
    )

    R = o3.rand_matrix()
    D_in = irreps_in.D_from_matrix(R)
    D_out = irreps_out.D_from_matrix(R)
    rotated_data = Data(pos=data.pos @ R.T, x=data.x @ D_in.T, batch=data.batch)

    pred = f(data)
    rotated_pred = f(rotated_data)

    assert (pred @ D_out.T - rotated_pred).abs().max() < 1e-10
