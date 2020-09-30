# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch

from e3nn import rs
from e3nn.non_linearities import GatedBlock, GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.linear import Linear


class DepthwiseConvolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, Rs_mid1, Rs_mid2, groups, convolution, linear=Linear, scalar_activation=swish, gate_activation=sigmoid, final_nonlinearity=True):
        """
        :param Rs_in:
        :param Rs_out:
        :param Rs_mid1:
        :param Rs_mid2:
        :param convolution: convolution operation that takes Rs_mid1, Rs_mid2
            e.g. convolution = lambda Rs_in, Rs_out: Convolution(Kernel(Rs_in, Rs_out, ConstantRadialModel))
        :param linear:
        :param scalar_activation:
        :param gated_activation:
        :param final_nonlinearity:
        """
        super().__init__()

        act_in = GatedBlock(
            groups * Rs_mid1, scalar_activation, gate_activation)
        self.lin_in = linear(Rs_in, act_in.Rs_in)
        self.act_in = act_in

        act_mid = GatedBlock(Rs_mid2, scalar_activation, gate_activation)
        self.conv = convolution(Rs_mid1, act_mid.Rs_in)
        self.act_mid = act_mid

        if final_nonlinearity:
            act_out = GatedBlock(Rs_out, scalar_activation, gate_activation)
            self.lin_out = linear(groups * Rs_mid2, act_out.Rs_in)
            self.act_out = act_out
        else:
            self.lin_out = linear(groups * Rs_mid2, Rs_out)
            self.act_out = None

        self.groups = groups

    def forward(self, features, *args, **kwargs):
        """
        :param features: tensor [..., point, channel]
        :return:         tensor [..., point, channel]
        """
        features = self.lin_in(features)
        features = self.act_in(features)

        features = self.conv(features, *args, **kwargs, groups=self.groups)
        features = self.act_mid(
            features.reshape(-1, rs.dim(self.act_mid.Rs_in))).reshape(*features.shape[:-1], -1)

        features = self.lin_out(features)
        if self.act_out is not None:
            features = self.act_out(features)

        return features


class DepthwiseConvolutionParity(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, Rs_mid1, Rs_mid2, groups, convolution, linear=Linear, scalar_activation=swish, gate_activation=sigmoid, final_nonlinearity=True):
        super().__init__()

        # Linear with GatedBlock
        scalars = [(mul, l, p) for (mul, l, p) in groups * Rs_mid1 if l == 0]
        act_scalars = [(mul, scalar_activation if p == 1 else tanh) for mul, l, p in scalars]

        nonscalars = [(mul, l, p) for (mul, l, p) in groups * Rs_mid1 if l > 0]
        gates = [(rs.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, gate_activation)]

        act_in = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
        self.lin_in = linear(Rs_in, act_in.Rs_in)
        self.act_in = act_in

        # Kernel with GatedBlock
        scalars = [(mul, l, p) for (mul, l, p) in Rs_mid2 if l == 0]
        act_scalars = [(mul, scalar_activation if p == 1 else tanh) for mul, l, p in scalars]

        nonscalars = [(mul, l, p) for (mul, l, p) in Rs_mid2 if l > 0]
        gates = [(rs.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, gate_activation)]

        act_mid = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
        self.conv = convolution(Rs_mid1, act_mid.Rs_in)
        self.act_mid = act_mid

        # Linear with or without GatedBlock
        if final_nonlinearity:
            scalars = [(mul, l, p) for (mul, l, p) in Rs_out if l == 0]
            act_scalars = [(mul, scalar_activation if p == 1 else tanh) for mul, l, p in scalars]

            nonscalars = [(mul, l, p) for (mul, l, p) in Rs_out if l > 0]
            gates = [(rs.mul_dim(nonscalars), 0, +1)]
            act_gates = [(-1, gate_activation)]

            act_out = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
            self.lin_out = linear(groups * Rs_mid2, act_out.Rs_in)
            self.act_out = act_out
        else:
            self.lin_out = linear(groups * Rs_mid2, Rs_out)
            self.act_out = None

        self.groups = groups

    def forward(self, features, *args, **kwargs):
        """
        :param features: tensor [..., point, channel]
        :return:         tensor [..., point, channel]
        """
        features = self.lin_in(features)
        features = self.act_in(features, groups=self.groups)

        features = self.conv(features, *args, **kwargs, groups=self.groups)
        features = self.act_mid(
            features.reshape(-1, rs.dim(self.act_mid.Rs_in))).reshape(*features.shape[:-1], -1)

        features = self.lin_out(features)
        if self.act_out is not None:
            features = self.act_out(features)

        return features
