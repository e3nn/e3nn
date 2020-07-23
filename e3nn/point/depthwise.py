# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch

from e3nn import rs
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import swish, sigmoid
from e3nn.linear import Linear


class DepthwiseConvolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, Rs_mid1, Rs_mid2, groups, convolution, linear=Linear, scalar_activation=swish, gate_activation=sigmoid, final_nonlinearity=True):
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
