# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn.non_linearities import GatedBlock, GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.s2 import S2Activation
from e3nn.point.operations import Convolution
from e3nn.radial import GaussianRadialModel
from e3nn.tensor_product import LearnableTensorSquare


class GatedConvNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_hidden, Rs_out, lmax, layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3,
                 feature_product=False, kernel=Kernel, convolution=Convolution):
        super().__init__()

        representations = [Rs_in]
        representations += [Rs_hidden] * layers
        representations += [Rs_out]

        RadialModel = partial(GaussianRadialModel, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=swish)

        K = partial(kernel, RadialModel=RadialModel, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

        def make_layer(Rs_in, Rs_out):
            if feature_product:
                tr1 = rs.TransposeToMulL(Rs_in)
                lts = LearnableTensorSquare(tr1.Rs_out, partial(o3.selection_rule, lmax=lmax))
                tr2 = torch.nn.Flatten(2)
                Rs = tr1.mul * lts.Rs_out
                act = GatedBlock(Rs_out, swish, sigmoid)
                conv = convolution(K(Rs, act.Rs_in))
                return torch.nn.ModuleList([torch.nn.Sequential(tr1, lts, tr2), conv, act])
            else:
                act = GatedBlock(Rs_out, swish, sigmoid)
                conv = convolution(K(Rs_in, act.Rs_in))
                return torch.nn.ModuleList([conv, act])

        self.layers = torch.nn.ModuleList([
            make_layer(Rs_layer_in, Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-2], representations[1:-1])
        ])

        self.layers.append(convolution(K(representations[-2], representations[-1])))
        self.feature_product = feature_product

    def forward(self, input, *args, **kwargs):
        output = input
        N = args[0].shape[-2]
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = N

        if self.feature_product:
            for ts, conv, act in self.layers[:-1]:
                output = ts(output)
                output = conv(output, *args, **kwargs)
                output = act(output)
        else:
            for conv, act in self.layers[:-1]:
                output = conv(output, *args, **kwargs)
                output = act(output)

        layer = self.layers[-1]
        output = layer(output, *args, **kwargs)
        return output


class GatedConvParityNetwork(torch.nn.Module):
    def __init__(self, Rs_in, mul, Rs_out, lmax, layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3,
                 feature_product=False, kernel=Kernel, convolution=Convolution):
        super().__init__()

        R = partial(GaussianRadialModel, max_radius=max_radius,
                    number_of_basis=number_of_basis, h=100,
                    L=radial_layers, act=swish)
        K = partial(kernel, RadialModel=R, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

        modules = []

        Rs = Rs_in
        for _ in range(layers):
            scalars = [(mul, l, p) for mul, l, p in [(mul, 0, +1), (mul, 0, -1)] if rs.haslinearpath(Rs, l, p)]
            act_scalars = [(mul, swish if p == 1 else tanh) for mul, l, p in scalars]

            nonscalars = [(mul, l, p) for l in range(1, lmax + 1) for p in [+1, -1] if rs.haslinearpath(Rs, l, p)]
            gates = [(rs.mul_dim(nonscalars), 0, +1)]
            act_gates = [(-1, sigmoid)]

            act = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
            conv = convolution(K(Rs, act.Rs_in))

            if feature_product:
                tr1 = rs.TransposeToMulL(act.Rs_out)
                lts = LearnableTensorSquare(tr1.Rs_out, partial(o3.selection_rule, lmax=lmax))
                tr2 = torch.nn.Flatten(2)
                act = torch.nn.Sequential(act, tr1, lts, tr2)
                Rs = tr1.mul * lts.Rs_out
            else:
                Rs = act.Rs_out

            block = torch.nn.ModuleList([conv, act])
            modules.append(block)

        self.firstlayers = torch.nn.ModuleList(modules)

        K = partial(K, allow_unused_inputs=True)
        self.conv = convolution(K(Rs, Rs_out))

    def forward(self, features, geometry, n_norm=None):
        if n_norm is None:
            n_norm = geometry.shape[1]

        for conv, act in self.firstlayers:
            features = conv(features, geometry, n_norm=n_norm)
            features = act(features)

        features = self.conv(features, geometry, n_norm=n_norm)
        return features


class S2Network(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, layers=3):
        super().__init__()

        Rs = rs.simplify(Rs_in)
        Rs_out = rs.simplify(Rs_out)

        self.layers = []

        for _ in range(layers):
            # tensor product: nonlinear and mixes the l's
            tp = rs.TensorSquare(Rs, selection_rule=partial(o3.selection_rule, lmax=lmax))

            # direct sum
            Rs = Rs + tp.Rs_out

            # linear: learned but don't mix l's
            Rs_act = [(1, l) for l in range(lmax + 1)]
            lin = Linear(Rs, mul * Rs_act, allow_unused_inputs=True)

            # s2 nonlinearity
            act = S2Activation(Rs_act, swish, res=20 * (lmax + 1))
            Rs = mul * act.Rs_out

            self.layers += [torch.nn.ModuleList([tp, lin, act])]

        self.layers = torch.nn.ModuleList(self.layers)

        def lfilter(l):
            return l in [j for _, j, _ in Rs_out]

        tp = rs.TensorSquare(Rs, selection_rule=partial(o3.selection_rule, lfilter=lfilter))
        Rs = Rs + tp.Rs_out
        lin = Linear(Rs, Rs_out, allow_unused_inputs=True)
        self.tail = torch.nn.ModuleList([tp, lin])

    def forward(self, x):
        for tp, lin, act in self.layers:
            xx = tp(x)
            x = torch.cat([x, xx], dim=-1)
            x = lin(x)

            x = x.reshape(*x.shape[:-1], -1, rs.dim(act.Rs_in))  # put multiplicity into batch
            x = act(x)
            x = x.reshape(*x.shape[:-2], -1)  # put back into representation

        tp, lin = self.tail
        xx = tp(x)
        x = torch.cat([x, xx], dim=-1)
        x = lin(x)

        return x
