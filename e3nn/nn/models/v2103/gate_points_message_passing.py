"""
>>> test()
"""
import torch
from e3nn import o3
from e3nn.nn import Gate

from .points_convolution import Convolution


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


class MessagePassing(torch.nn.Module):
    r"""

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

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_hidden,
        irreps_node_output,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
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
            ]).simplify()
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ])
            ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
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
                fc_neurons,
                num_neighbors
            )
        )

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features


def test():
    from torch_cluster import radius_graph
    from e3nn.util.test import assert_equivariant, assert_auto_jitable

    mp = MessagePassing(
        irreps_node_input="0e",
        irreps_node_hidden="0e + 1e",
        irreps_node_output="1e",
        irreps_node_attr="0e + 1e",
        irreps_edge_attr="1e",
        layers=3,
        fc_neurons=[2, 100],
        num_neighbors=3.0,
    )

    num_nodes = 4
    node_pos = torch.randn(num_nodes, 3)
    edge_index = radius_graph(node_pos, 3.0)
    edge_src, edge_dst = edge_index
    num_edges = edge_index.shape[1]
    edge_attr = node_pos[edge_index[0]] - node_pos[edge_index[1]]

    node_features = torch.randn(num_nodes, 1)
    node_attr = torch.randn(num_nodes, 4)
    edge_scalars = torch.randn(num_edges, 2)

    assert mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars).shape == (num_nodes, 3)

    assert_equivariant(
        mp,
        irreps_in=[mp.irreps_node_input, mp.irreps_node_attr, None, None, mp.irreps_edge_attr, None],
        args_in=[node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars],
        irreps_out=[mp.irreps_node_output],
    )

    assert_auto_jitable(mp.layers[0].first)
