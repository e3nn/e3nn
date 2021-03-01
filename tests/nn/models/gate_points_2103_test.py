import torch
from torch_cluster import radius_graph

from e3nn import o3
from e3nn.nn.models.gate_points_2103 import Network
from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_simple():
    net = Network(
        irreps_node_input="0e",
        irreps_node_hidden="0e + 1e",
        irreps_node_output="1e",
        irreps_node_attr="0e + 1e",
        irreps_edge_attr="1e",
        layers=3,
        number_of_edge_scalars=2,
        fc_layers=1,
        fc_neurons=100,
        num_neighbors=3.0,
    )

    num_nodes = 4
    node_pos = torch.randn(num_nodes, 3)
    edge_index = radius_graph(node_pos, 3.0)
    num_edges = edge_index.shape[1]
    edge_attr = node_pos[edge_index[0]] - node_pos[edge_index[1]]

    data = {
        'node_input': torch.randn(num_nodes, 1),
        'node_attr': torch.randn(num_nodes, 4),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_scalars': torch.randn(num_edges, 2),
    }

    assert net(data).shape == (num_nodes, 3)

    def wrapper(edge_index, edge_attr, node_input, node_attr, edge_scalars):
        data = {
            'node_input': node_input,
            'node_attr': node_attr,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_scalars': edge_scalars,
        }

        return net(data)

    assert_equivariant(
        wrapper,
        irreps_in=[None, net.irreps_edge_attr, net.irreps_node_input, net.irreps_node_attr, net.number_of_edge_scalars * o3.Irrep('0e')],
        args_in=[data['edge_index'], data['edge_attr'], data['node_input'], data['node_attr'], data['edge_scalars']],
        irreps_out=[net.irreps_node_output],
    )

    assert_auto_jitable(net.layers[0].first)
