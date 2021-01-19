"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2020
"""
import math
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace


class Convolution(torch.nn.Module):
    def __init__(
            self,
            irreps_in,
            irreps_node_attr,
            irreps_edge_attr,
            irreps_out,
            number_of_basis,
            radial_layers,
            radial_neurons,
            num_neighbors
        ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.si = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], torch.relu)
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        si = self.si(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x)).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)
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
    r"""equivariant neural network

    Parameters
    ----------
    irreps_in : `o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `o3.Irreps`
        representation of the hidden features

    irreps_out : `o3.Irreps`
        representation of the output features

    irreps_node_attr : `o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials

    layers : int
        number of gates (non linearities)

    max_radius : float
        maximum radius for the convolution

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes at a distance ``max_radius``

    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
            self,
            irreps_in,
            irreps_hidden,
            irreps_out,
            irreps_node_attr,
            irreps_edge_attr,
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

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else irreps_node_attr
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        irreps = self.irreps_in if self.irreps_in is not None else self.irreps_edge_attr

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
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_nonscalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_nonscalars])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_nonscalars  # non-scalars
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr if self.irreps_node_attr is not None else "1e",
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr if self.irreps_node_attr is not None else "1e",
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def forward(self, data) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data`
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = None

        edge_src, edge_dst = radius_graph(data.pos, self.max_radius, batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, normalization='component', normalize=True)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis).mul(self.number_of_basis**0.5)
        edge_attr = smooth_transition(edge_length / self.max_radius)[:, None] * edge_sh

        if self.irreps_in is not None:
            x = data.x
        else:
            x = scatter(edge_attr, edge_dst, dim=0, dim_size=len(data.pos)).div(self.num_neighbors**0.5)

        if self.irreps_node_attr is not None:
            z = data.z
        else:
            z = x.new_ones(x.shape[0], 1)

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
