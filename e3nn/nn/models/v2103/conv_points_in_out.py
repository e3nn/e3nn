r"""example of a graph convolution when the input and output nodes are different

>>> test()
"""
import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode


@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `Irreps`
        representation of the input node features

    irreps_node_output : `Irreps` or None
        representation of the output node features

    irreps_node_attr_input : `Irreps`
        representation of the input node attributes

    irreps_node_attr_output : `Irreps`
        representation of the output node attributes

    irreps_edge_attr : `Irreps`
        representation of the edge attributes

    num_edge_scalar_attr : int
        number of scalar (0e) attributes of the edge used to feed the FC network

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        irreps_node_attr_input,
        irreps_node_attr_output,
        irreps_edge_attr,
        num_edge_scalar_attr,
        radial_layers,
        radial_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr_input = o3.Irreps(irreps_node_attr_input)
        self.irreps_node_attr_output = o3.Irreps(irreps_node_attr_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr_input, self.irreps_node_input)

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
            [num_edge_scalar_attr] + radial_layers * [radial_neurons] + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr_output, self.irreps_node_output)

    def forward(self, node_input, node_attr_input, node_attr_output, edge_src, edge_dst, edge_attr, edge_scalar_attr) -> torch.Tensor:
        weight = self.fc(edge_scalar_attr)

        node_input = self.lin1(node_input, node_attr_input)

        edge_features = self.tp(node_input[edge_src], edge_attr, weight)
        node_output = scatter(edge_features, edge_dst, dim=0, dim_size=node_attr_output.shape[0])
        node_output.div_(self.num_neighbors**0.5)

        return self.lin2(node_output, node_attr_output)


def test():
    from torch_cluster import radius
    from e3nn.math import soft_one_hot_linspace

    conv = Convolution(
        irreps_node_input='0e + 1e',
        irreps_node_output='0e + 1e',
        irreps_node_attr_input='2x0e',
        irreps_node_attr_output='3x0e',
        irreps_edge_attr='0e + 1e',
        num_edge_scalar_attr=4,
        radial_layers=1,
        radial_neurons=50,
        num_neighbors=3.0,
    )

    pos_in = torch.randn(5, 3)
    pos_out = torch.randn(2, 3)

    node_input = torch.randn(5, 4)
    node_attr_input = torch.randn(5, 2)
    node_attr_output = torch.randn(2, 3)

    edge_src, edge_dst = radius(pos_out, pos_in, r=2.0)
    edge_vec = pos_in[edge_src] - pos_out[edge_dst]
    edge_attr = o3.spherical_harmonics([0, 1], edge_vec, True)
    edge_scalar_attr = soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.0, end=2.0, number=4, basis='smooth_finite', cutoff=True)

    conv(node_input, node_attr_input, node_attr_output, edge_src, edge_dst, edge_attr, edge_scalar_attr)
