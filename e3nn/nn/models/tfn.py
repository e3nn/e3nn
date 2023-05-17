"""Tensor Field Networks (TFN)

Implementation of the TFN model using e3nn
"""
import math
from typing import Dict, Union

import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""Equivariant convolution for Tensor Field Networks (TFN)

    Based on https://github.com/e3nn/e3nn/issues/264
    node_input: node's feature data that flow in the network with message passing
    node_attr: node's features that are independent of the message passing step (like atom type)
    edge_attr: edge's features that are independent of the message passing step 
        (like sh of the relative position vector)
    edge_features (edge_scalars):
        scalar quantities that are independent of the message passing step used to feed
        the MLP that generates the weights for the TP (usually function of the relative distance)

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_edge_features : int
        number of scalar (0e) features of the edge used to feed the FC network

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        average number of nodes convolved over, used for normalization during scatter()
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_node_attr: str,
        irreps_edge_attr: str,
        irreps_out: str,
        number_of_edge_features: str,
        radial_layers: str,
        radial_neurons: str,
        num_neighbors: int,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # Self connection layer
        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]
        
        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_edge_features] + radial_layers * [radial_neurons] + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self, node_input, node_attr, edge_index, edge_attr, edge_scalars) -> torch.Tensor:
        '''Forward pass of the TFN model
        
        sc, lin1, lin2: Prepares the input using mixing of
        - node input (features that flow during message passing)
        - node attributes (features that are independent of the message passing step)

        tp: Performs tensor product between x and edge_attr
        (edge_attr is Equation 2 in TFN paper)
        '''
        weight = self.fc(edge_scalars)

        # decompose edge_index into edge_src and edge_dst
        edge_src, edge_dst = edge_index[0], edge_index[1]
        x = node_input

        # sc: Skip connection, will be used in the end
        s = self.sc(x, node_attr)

        # Perform message passing
        # 1. Prepare x by mixing with node attributes (no mixing here)
        x = self.lin1(x, node_attr)

        # 2. Prepare message using x, edge attributes and weights
        edge_features = self.tp(x[edge_src], edge_attr, weight)

        # 3. Perform the scatter / message passing operation
        x = scatter(
            edge_features, edge_dst, dim=0, dim_size=x.shape[0]
        ).div(self.num_neighbors**0.5)

        # Post processing
        x = self.lin2(x, node_attr)

        # Skip connection
        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


def smooth_cutoff(x):
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
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class TFN(torch.nn.Module):
    r"""Tensor Field Network (TFN)

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_out : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
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
        irreps_in: str,
        irreps_hidden: str,
        irreps_out: str,
        irreps_node_attr: str,
        irreps_edge_attr: str,
        layers: int,
        max_radius: float,
        number_of_basis,
        num_neighbors: float,
        num_nodes: int,
        radial_layers: int = 1,
        radial_neurons: int = 50,
        reduce_output: bool=True,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None
        number_of_edge_features = number_of_basis

        self.ext_z = ExtractIr(self.irreps_node_attr, "0e")

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

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
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            self.layers.append(Compose(conv, gate))
            
            # irreps for next layer
            irreps = gate.irreps_out

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass of the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        
        # spherical harmonics based on normalized relative positions
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")

        # get distances between nodes and embed them into a basis
        edge_length = edge_vec.norm(dim=1)
        
        # TODO: Ask about this step (not present in original TFN)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius, number=self.number_of_basis, basis="gaussian", cutoff=False
        ).mul(self.number_of_basis**0.5)

        # Combination of distances and spherical harmonics
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        # optional bit: need not be present
        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for layer in self.layers:
            x = layer(x, z, edge_index, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x
