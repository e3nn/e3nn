from typing import Dict, Union

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from .gate_points_message_passing import MessagePassing


class SimpleNetwork(torch.nn.Module):
    def __init__(
            self,
            irreps_in,
            irreps_out,
            max_radius,
            num_neighbors,
            num_nodes,
            mul=50,
            layers=3,
            lmax=2,
                ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            base='cosine',  # the cosine basis with endpoint = True goes to zero at max_radius
            endpoint=False,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = data['pos'].new_ones(data['pos'].shape[0], 1)

        node_outputs = self.mp(data['x'], node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)


class MoreAdvancedNetwork(torch.nn.Module):
    def __init__(
            self,
            irreps_node_input,
            irreps_node_output,
            irreps_node_hidden,
            irreps_node_attr,
            irreps_edge_attr,
            max_radius,
            num_neighbors,
            num_nodes,
            layers,
            lmax_sh,
                ) -> None:
        super().__init__()
        # TODO

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        # TODO
        pass
