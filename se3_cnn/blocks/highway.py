# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3BNConvolution, SE3ConvolutionBN
from se3_cnn.non_linearities import ScalarActivation
from se3_cnn import SO3


class HighwayBlock(torch.nn.Module):
    def __init__(self,
                 repr_in, repr_out, size, radial_window_dict, # kernel params
                 activation=None, stride=1, padding=0, # conv/nonlinearity params
                 batch_norm_momentum=0.1, batch_norm_mode='normal', batch_norm_before_conv=True): # batch norm params
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param repr_out: same but for the output
        :param int size: the filters are cubes of dimension = size x size x size
        :param radial_window_dict: contains both radial window function and the keyword arguments for the radial window function
        :param activation: function like for instance torch.nn.functional.relu
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
        if activation is not None:
            Rs_out += [(sum(repr_out[1:]), SO3.repr1)]

        self.bn_conv = (SE3BNConvolution if batch_norm_before_conv else SE3ConvolutionBN)(
            Rs_in=Rs_in,
            Rs_out=Rs_out,
            size=size,
            radial_window_dict=radial_window_dict,
            stride=stride,
            padding=padding,
            momentum=batch_norm_momentum,
            mode=batch_norm_mode)

        if activation is not None:
            self.act = ScalarActivation(
                [(mul * (2 * n + 1), n == 0) for n, mul in enumerate(repr_out)] + [(sum(repr_out[1:]), True)], activation)
        else:
            self.act = None

    def forward(self, x):  # pylint: disable=W
        y = self.bn_conv(x)

        if self.act is None:
            return y

        y = self.act(y)

        if sum(self.repr_out[1:]) == 0:
            return y

        nbatch = y.size(0)
        nx = y.size(2)
        ny = y.size(3)
        nz = y.size(4)

        begin_y = self.repr_out[0]
        begin_u = sum(mul * (2 * n + 1) for n, mul in enumerate(self.repr_out))

        zs = []
        if self.repr_out[0] != 0:
            zs.append(y[:, :self.repr_out[0]])

        for n, mul in enumerate(self.repr_out):
            if n == 0:
                continue
            if mul == 0:
                continue
            dim = 2 * n + 1

            field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, x, y, z]
            field_y = field_y.contiguous()
            field_y = field_y.view(nbatch, mul, dim, nx, ny, nz)  # [batch, feature, repr, x, y, z]

            field_u = y[:, begin_u: begin_u + mul]  # [batch, feature, x, y, z]
            field_u = field_u.contiguous()
            field_u = field_u.view(nbatch, mul, 1, nx, ny, nz)  # [batch, feature, repr, x, y, z]

            field = field_y * field_u  # [batch, feature, repr, x, y, z]
            field = field.view(nbatch, mul * dim, nx, ny, nz)  # [batch, feature * repr, x, y, z]

            zs.append(field)

            begin_y += mul * dim
            begin_u += mul

        return torch.cat(zs, dim=1)
