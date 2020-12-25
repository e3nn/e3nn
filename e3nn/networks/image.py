# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name, abstract-method
import torch
from e3nn import rs
from e3nn.image.convolution import Convolution
from e3nn.image.filter import LowPassFilter
from e3nn.non_linearities import GatedBlock, GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.s2 import S2Activation
from e3nn.tensor_product import LearnableTensorSquare


class ImageGatedConvNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_hidden, Rs_out, lmax, size=5, layers=3):
        super().__init__()

        representations = [Rs_in]
        representations += [Rs_hidden] * layers
        representations += [Rs_out]

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, swish, sigmoid)
            conv = Convolution(
                Rs_in, act.Rs_in, size, lmax=lmax, fuzzy_pixels=True, padding=size // 2)
            return torch.nn.Sequential(conv, act)

        self.layers = torch.nn.Sequential(*[
            make_layer(Rs_layer_in, Rs_layer_out)
            for Rs_layer_in, Rs_layer_out in zip(representations[:-2], representations[1:-1])
        ] + [
            Convolution(representations[-2], representations[-1], size, lmax=lmax, fuzzy_pixels=True)
        ])

    def forward(self, input):
        """
        :param input: tensor of shape [batch, x, y, z, channel]
        """
        return self.layers(input)


class ImageGatedConvParityNetwork(torch.nn.Module):
    def __init__(self, Rs_in, mul, Rs_out, lmax, size=5, layers=3):
        super().__init__()

        modules = []

        Rs = rs.convention(Rs_in)
        for _ in range(layers):
            scalars = [(mul, l, p) for mul, l, p in [(mul, 0, +1),
                                                     (mul, 0, -1)] if rs.haslinearpath(Rs, l, p)]
            act_scalars = [(mul, swish if p == 1 else tanh)
                           for mul, l, p in scalars]

            nonscalars = [(mul, l, p) for l in range(1, lmax + 1)
                          for p in [+1, -1] if rs.haslinearpath(Rs, l, p)]
            gates = [(rs.mul_dim(nonscalars), 0, +1)]
            if rs.haslinearpath(Rs, 0, +1):
                gates = [(rs.mul_dim(nonscalars), 0, +1)]
                act_gates = [(-1, sigmoid)]
            else:
                gates = [(rs.mul_dim(nonscalars), 0, -1)]
                act_gates = [(-1, tanh)]

            act = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
            conv = Convolution(Rs, act.Rs_in, size, lmax=lmax, fuzzy_pixels=True, padding=size // 2)

            Rs = act.Rs_out

            block = torch.nn.Sequential(conv, act)
            modules.append(block)

        modules += [Convolution(Rs, Rs_out, size, lmax=lmax, fuzzy_pixels=True, padding=size // 2, allow_unused_inputs=True)]
        self.layers = torch.nn.Sequential(*modules)

    def forward(self, input):
        """
        :param input: tensor of shape [batch, x, y, z, channel]
        """
        return self.layers(input)


class ImageS2Network(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, size=5, layers=3):
        super().__init__()

        Rs = rs.simplify(Rs_in)
        Rs_out = rs.simplify(Rs_out)
        Rs_act = list(range(lmax + 1))

        self.mul = mul
        self.layers = []

        for _ in range(layers):
            conv = Convolution(
                Rs, mul * Rs_act, size, lmax=lmax, fuzzy_pixels=True, padding=size // 2)

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

            # put multiplicity into batch
            x = x.reshape(*x.shape[:-1], self.mul, rs.dim(act.Rs_in))
            x = act(x)
            # put back into representation
            x = x.reshape(*x.shape[:-2], self.mul * rs.dim(act.Rs_out))

            x = pool(x)

        x = self.tail(x)
        return x
