# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3BNConvolution
from se3_cnn.non_linearities import NormSoftplus
from se3_cnn import SO3
from se3_cnn.dropout import SE3Dropout


class NormBlock(torch.nn.Module):
    def __init__(self,
                 repr_in, repr_out, size, radial_window,  # kernel params
                 activation=None, activation_bias_min=0.5, activation_bias_max=2,
                 stride=1, padding=0, capsule_dropout_p=None,  # conv/nonlinearity params
                 normalization=None, batch_norm_momentum=0.1):  # batch norm params
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param repr_out: same but for the output
        :param int size: the filters are cubes of dimension = size x size x size
        :param radial_window: radial window function
        :param activation: function like for instance torch.nn.functional.relu
        :param activation_bias_min Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        :param activation_bias_max Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        :param int stride: stride of the convolution (for torch.nn.functional.conv3d)
        :param int padding: padding of the convolution (for torch.nn.functional.conv3d)
        :param float conv_dropout_p: Convolution dropout probability
        :param str normalization: "batch", "group", "instance" or None
        :param float batch_norm_momentum: batch normalization momentum (put it to zero to disable the batch normalization)
        '''
        super().__init__()
        self.repr_out = repr_out

        irreducible_repr = [SO3.repr1, SO3.repr3, SO3.repr5, SO3.repr7, SO3.repr9, SO3.repr11, SO3.repr13, SO3.repr15]

        Rs_in = list(zip(repr_in, irreducible_repr))
        Rs_out = list(zip(repr_out, irreducible_repr))

        if normalization is None:
            Convolution = SE3Convolution
        if normalization is "batch":
            Convolution = partial(SE3BNConvolution, momentum=batch_norm_momentum)
        if normalization is "group":
            Convolution = SE3GNConvolution
        if normalization == "instance":
            Convolution = partial(SE3GNConvolution, Rs_gn=[(1, 2 * n + 1) for n, mul in enumerate(repr_in) for _ in range(mul)])

        self.conv = Convolution(
            Rs_in=Rs_in,
            Rs_out=Rs_out,
            size=size,
            radial_window=radial_window,
            stride=stride,
            padding=padding,
            momentum=batch_norm_momentum,
        )

        if capsule_dropout_p is not None:
            Rs_out_without_gate = [(mul, 2 * n + 1) for n, mul in enumerate(repr_out)]  # Rs_out without gates
            self.dropout = SE3Dropout(Rs_out_without_gate, capsule_dropout_p)
        else:
            self.dropout = None

        self.act = None
        if activation is not None:
            capsule_dims = [2 * n + 1 for n, mul in enumerate(repr_out) for i in
                            range(mul)]  # list of capsule dimensionalities
            self.act = NormSoftplus(capsule_dims,
                                    scalar_act=activation,
                                    bias_min=activation_bias_min,
                                    bias_max=activation_bias_max)

    def forward(self, x):  # pylint: disable=W
        y = self.bn_conv(x)

        if self.act is not None:
            y = self.act(y)

        # dropout
        if self.dropout is not None:
            y = self.dropout(y)

        return y
