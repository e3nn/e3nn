# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3BNConvolution, SE3ConvolutionBN
from se3_cnn.non_linearities import NormSoftplus
from se3_cnn import SO3


class NormBlock(torch.nn.Module):
    def __init__(self,
                 repr_in, repr_out, size, radial_window_dict,  # kernel params
                 activation=None, activation_bias_min=0.5, activation_bias_max=2,
                 stride=1, padding=0,  # conv/nonlinearity params
                 batch_norm_momentum=0.1, batch_norm_mode='normal', batch_norm_before_conv=True):  # batch norm params
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param repr_out: same but for the output
        :param int size: the filters are cubes of dimension = size x size x size
        :param radial_window_dict: contains both radial window function and the keyword arguments for the radial window function
        :param activation: function like for instance torch.nn.functional.relu
        :param activation_bias_min Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        :param activation_bias_max Activation bias is initialized uniformly from [activation_bias_min, activation_bias_max]
        :param int stride: stride of the convolution (for torch.nn.functional.conv3d)
        :param int padding: padding of the convolution (for torch.nn.functional.conv3d)
        :param float batch_norm_momentum: batch normalization momentum (put it to zero to disable the batch normalization)
        :param batch_norm_mode: the mode of the batch normalization
        :param bool batch_norm_before_conv: perform the batch normalization before or after the convolution
        '''
        super().__init__()
        self.repr_out = repr_out

        irreducible_repr = [SO3.repr1, SO3.repr3, SO3.repr5, SO3.repr7, SO3.repr9, SO3.repr11, SO3.repr13, SO3.repr15]

        Rs_in = list(zip(repr_in, irreducible_repr))
        Rs_out = list(zip(repr_out, irreducible_repr))

        self.bn_conv = (SE3BNConvolution if batch_norm_before_conv else SE3ConvolutionBN)(
            Rs_in=Rs_in,
            Rs_out=Rs_out,
            size=size,
            radial_window_dict=radial_window_dict,
            stride=stride,
            padding=padding,
            momentum=batch_norm_momentum,
            mode=batch_norm_mode)

        capsule_dims = [2 * n + 1 for n, mul in enumerate(repr_out) for i in
                        range(mul)]  # list of capsule dimensionalities
        self.act = NormSoftplus(capsule_dims,
                                scalar_act=activation,
                                bias_min=activation_bias_min,
                                bias_max=activation_bias_max)

    def forward(self, x):  # pylint: disable=W
        y = self.bn_conv(x)
        y = self.act(y)

        return y
