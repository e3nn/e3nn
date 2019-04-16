# pylint: disable=C,R,E1101
from functools import partial
import torch
from se3cnn.convolution import SE3PointConvolution
from se3cnn.non_linearities import NormSoftplus
from se3cnn import kernel


class PointNormBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, radii, activation=None, activation_bias_min=0.5, activation_bias_max=2, **kwargs):
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param repr_out: same but for the output
        :param radii: radii for basis functions
        :param activation: function like for instance torch.nn.functional.relu
        :param activation_bias_min Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        :param activation_bias_max Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        '''
        super().__init__()
        self.repr_out = repr_out

        Rs_in = [(m, l) for l, m in enumerate(repr_in)]
        Rs_out = [(m, l) for l, m in enumerate(repr_out)]

        Convolution = SE3PointConvolution

        self.conv = Convolution(
            Rs_in=Rs_in,
            Rs_out=Rs_out,
            radii=radii,
            **kwargs,
        )

        self.act = None
        if activation is not None:
            capsule_dims = [2 * n + 1 for n, mul in enumerate(repr_out) for i in
                            range(mul)]  # list of capsule dimensionalities
            self.act = NormSoftplus(capsule_dims,
                                    scalar_act=activation,
                                    bias_min=activation_bias_min,
                                    bias_max=activation_bias_max)

    def forward(self, x, diff_M, relative_mask=None):  # pylint: disable=W
        y = self.conv(x, diff_M, relative_mask)

        if self.act is not None:
            y = self.act(y)

        return y
