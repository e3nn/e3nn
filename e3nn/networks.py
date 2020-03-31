# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
from functools import partial

import torch

import e3nn.o3 as o3
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid, swish
from e3nn.point.operations import Convolution
from e3nn.radial import GaussianRadialModel
from e3nn.tensor_product import TensorProduct
from e3nn.linear import Linear

torch.set_default_dtype(torch.float64)


class GatedConvNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_hidden, Rs_out, lmax, layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3,
                 feature_product=False):
        super().__init__()

        representations = [Rs_in]
        representations += [Rs_hidden] * layers
        representations += [Rs_out]

        RadialModel = partial(GaussianRadialModel, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=swish)

        K = partial(Kernel, RadialModel=RadialModel, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

        def make_layer(Rs_in, Rs_out):
            if feature_product:
                tp = TensorProduct(Rs_in, Rs_in, selection_rule=partial(o3.selection_rule, lmax=lmax))
                lin = Linear(tp.Rs_out, Rs_in)
            act = GatedBlock(Rs_out, swish, sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)
            if feature_product:
                return torch.nn.ModuleList([tp, lin, conv, act])
            return torch.nn.ModuleList([conv, act])

        self.layers = torch.nn.ModuleList([
            make_layer(Rs_layer_in, Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-2], representations[1:-1])
        ])

        self.layers.append(Convolution(K, representations[-2], representations[-1]))
        self.feature_product = feature_product

    def forward(self, input, geometry):
        output = input
        _, N, _ = geometry.shape

        if self.feature_product:
            for tp, lin, conv, act in self.layers[:-1]:
                output = tp(output)
                output = lin(output)
                output = conv(output, geometry, n_norm=N)
                output = act(output)
        else:
            for conv, act in self.layers[:-1]:
                output = conv(output, geometry, n_norm=N)
                output = act(output)

        layer = self.layers[-1]
        output = layer(output, geometry, n_norm=N)

        return output
