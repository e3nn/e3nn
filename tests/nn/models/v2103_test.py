import torch
from torch_cluster import radius_graph

from e3nn.nn.models.v2103 import MessagePassing, SimpleNetwork
from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_message_passing():
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


def test_simple_network():
    net = SimpleNetwork(
        "3x0e + 2x1o",
        "4x0e + 1x1o",
        max_radius=2.0,
        num_neighbors=3.0,
        num_nodes=5.0
    )

    net({
        'pos': torch.randn(5, 3),
        'x': net.irreps_in.randn(5, -1)
    })
