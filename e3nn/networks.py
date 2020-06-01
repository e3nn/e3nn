# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock, GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.s2 import S2Activation
from e3nn.point.operations import Convolution
from e3nn.image.convolution import Convolution as ImageConvolution
from e3nn.image.filter import LowPassFilter
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
                lts = LearnableTensorSquare(tr1.Rs_out, list(range(lmax + 1)), allow_change_output=True)
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
                lts = LearnableTensorSquare(tr1.Rs_out, [(1, l, p) for l in range(lmax + 1) for p in [-1, 1]], allow_change_output=True)
                tr2 = torch.nn.Flatten(2)
                act = torch.nn.Sequential(act, tr1, lts, tr2)
                Rs = tr1.mul * lts.Rs_out
            else:
                Rs = act.Rs_out

            block = torch.nn.ModuleList([conv, act])
            modules.append(block)

        self.layers = torch.nn.ModuleList(modules)

        K = partial(K, allow_unused_inputs=True)
        self.layers.append(convolution(K(Rs, Rs_out)))
        self.feature_product = feature_product

    def forward(self, input, *args, **kwargs):
        output = input
        N = args[0].shape[-2]
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = N

        for conv, act in self.layers[:-1]:
            output = conv(output, *args, **kwargs)
            output = act(output)

        layer = self.layers[-1]
        output = layer(output, *args, **kwargs)
        return output


class S2ConvNetwork(torch.nn.Module):
    def __init__(self, Rs_in, mul, Rs_out, lmax, layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3,
                 kernel=Kernel, convolution=Convolution):
        super().__init__()

        Rs_hidden = [(1, l, (-1)**l) for i in range(mul) for l in range(lmax + 1)]
        representations = [Rs_in]
        representations += [Rs_hidden] * layers
        representations += [Rs_out]

        RadialModel = partial(GaussianRadialModel, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=swish)

        K = partial(kernel, RadialModel=RadialModel, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

        def make_layer(Rs_in, Rs_out):
            act = S2Activation([(1, l, (-1)**l) for l in range(lmax + 1)], sigmoid, lmax_out=lmax, res=20 * (lmax + 1))
            conv = convolution(K(Rs_in, Rs_out))
            return torch.nn.ModuleList([conv, act])

        self.layers = torch.nn.ModuleList([
            make_layer(Rs_layer_in, Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-2], representations[1:-1])
        ])

        self.layers.append(convolution(K(representations[-2], representations[-1])))
        self.mul = mul
        self.lmax = lmax

    def forward(self, input, *args, **kwargs):
        output = input
        N = args[0].shape[-2]
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = N

        for conv, act in self.layers[:-1]:
            output = conv(output, *args, **kwargs)
            shape = list(output.shape)
            # Split multiplicities into new batch
            output = output.reshape(shape[:-1] + [self.mul, (self.lmax + 1) ** 2])
            output = act(output)
            output = output.reshape(shape)

        layer = self.layers[-1]
        output = layer(output, *args, **kwargs)
        return output


class S2Network(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, layers=3):
        super().__init__()

        Rs = rs.simplify(Rs_in)
        Rs_out = rs.simplify(Rs_out)
        Rs_act = list(range(lmax + 1))

        self.layers = []

        for _ in range(layers):
            lin = LearnableTensorSquare(Rs, mul * Rs_act, linear=True, allow_zero_outputs=True)

            # s2 nonlinearity
            act = S2Activation(Rs_act, swish, res=20 * (lmax + 1))
            Rs = mul * act.Rs_out

            self.layers += [torch.nn.ModuleList([lin, act])]

        self.layers = torch.nn.ModuleList(self.layers)

        self.tail = LearnableTensorSquare(Rs, Rs_out)

    def forward(self, x):
        for lin, act in self.layers:
            x = lin(x)

            x = x.reshape(*x.shape[:-1], -1, rs.dim(act.Rs_in))  # put multiplicity into batch
            x = act(x)
            x = x.reshape(*x.shape[:-2], -1)  # put back into representation

        x = self.tail(x)
        return x


class ImageS2Network(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, size=5, layers=3):
        super().__init__()

        Rs = rs.simplify(Rs_in)
        Rs_out = rs.simplify(Rs_out)
        Rs_act = list(range(lmax + 1))

        self.mul = mul
        self.layers = []

        for _ in range(layers):
            conv = ImageConvolution(Rs, mul * Rs_act, size, lmax=lmax, fuzzy_pixels=True, padding=size // 2)

            # s2 nonlinearity
            act = S2Activation(Rs_act, swish, res=60)
            Rs = mul * act.Rs_out

            pool = LowPassFilter(scale=2.0, stride=2)

            self.layers += [torch.nn.ModuleList([conv, act, pool])]

        self.layers = torch.nn.ModuleList(self.layers)
        self.tail = LearnableTensorSquare(Rs, Rs_out)

    def forward(self, x):
        """
        :param x: [batch, x, y, z, channel_in]
        :return: [batch, x, y, z, channel_out]
        """
        for conv, act, pool in self.layers:
            x = conv(x)

            x = x.reshape(*x.shape[:-1], self.mul, rs.dim(act.Rs_in))  # put multiplicity into batch
            x = act(x)
            x = x.reshape(*x.shape[:-2], self.mul * rs.dim(act.Rs_out))  # put back into representation

            x = pool(x)

        x = self.tail(x)
        return x
