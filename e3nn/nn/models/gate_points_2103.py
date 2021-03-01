"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of march 2021
"""
import math
from typing import Dict, Union

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode


@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `Irreps`
        representation of the input node features

    irreps_node_attr : `Irreps`
        representation of the node attributes

    irreps_edge_attr : `Irreps`
        representation of the edge attributes

    irreps_node_output : `Irreps` or None
        representation of the output node features

    number_of_edge_scalars : int
        number of scalar (0e) features of the edge used to feed the FC network

    fc_layers : int
        number of hidden layers in the fully connected network

    fc_neurons : int
        number of neurons in the hidden layers of the fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
            self,
            irreps_node_input,
            irreps_node_attr,
            irreps_edge_attr,
            irreps_node_output,
            number_of_edge_scalars,
            fc_layers,
            fc_neurons,
            num_neighbors
                ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_edge_scalars] + fc_layers * [fc_neurons] + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_scalars = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_scalars, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


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
    irreps_node_input : `Irreps`
        representation of the input features

    irreps_node_hidden : `Irreps`
        representation of the hidden features

    irreps_node_output : `Irreps`
        representation of the output features

    irreps_node_attr : `Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `Irreps`
        representation of the edge attributes

    layers : int
        number of gates (non linearities)

    number_of_edge_scalars : int
        number of scalar (0e) features of the edge used to feed the FC networks

    fc_layers : int
        number of hidden layers in the fully connected networks

    fc_neurons : int
        number of neurons in the hidden layers of the fully connected networks
    """
    def __init__(
            self,
            irreps_node_input,
            irreps_node_hidden,
            irreps_node_output,
            irreps_node_attr,
            irreps_edge_attr,
            layers,
            number_of_edge_scalars,
            fc_layers,
            fc_neurons,
            num_neighbors,
                ) -> None:
        super().__init__()
        self.number_of_edge_scalars = number_of_edge_scalars
        self.num_neighbors = num_neighbors

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        irreps_node = self.irreps_node_input

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ])
            irreps_nonscalars = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ])
            ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_nonscalars])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_nonscalars  # non-scalars
            )
            conv = Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                self.number_of_edge_scalars,
                fc_layers,
                fc_neurons,
                num_neighbors
            )
            irreps_node = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_node_output,
                self.number_of_edge_scalars,
                fc_layers,
                fc_neurons,
                num_neighbors
            )
        )

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``node_input`` the input features of the nodes
            - ``node_attr`` the attributes of the nodes, for instance the atom type
            - ``edge_index`` the graph: edge_src and edge_dst
            - ``edge_attr``
            - ``edge_scalars``
        """
        node_features = data['node_input']
        node_attr = data['node_attr']
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_attr = data['edge_attr']
        edge_scalars = data['edge_scalars']

        for lay in self.layers:
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features
