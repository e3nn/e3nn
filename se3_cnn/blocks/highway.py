# pylint: disable=C,R,E1101
import torch
from se3_cnn.bn_conv import SE3BNConvolution
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn import SO3

class HighwayBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, non_linearities, size, radial_amount, stride=1, padding=0, batch_norm_momentum=0.1):
        '''
        :param: repr_in: tuple with multiplicities of repr. (1, 3, 5)
        :param: repr_out: same but for the output
        :param: non_linearities: boolean, enable or not relu
        :param: size: the filters are cubes of dimension = size x size x size
        :param: radial_amount: number of radial discretization
        :param: stride: stride of the convolution (for torch.nn.functional.conv3d)
        :param: padding: padding of the convolution (for torch.nn.functional.conv3d)
        :param: batch_norm_momentum: batch normalization momentum
        '''
        super().__init__()
        self.repr_out = repr_out

        Rs_out = [(repr_out[1] + repr_out[2], SO3.repr1)] if non_linearities else []

        self.bn_conv = SE3BNConvolution(
            size=size,
            radial_amount=radial_amount,
            Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5)],
            Rs_out=[(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)] + Rs_out,
            stride=stride,
            padding=padding,
            momentum=batch_norm_momentum,
            mode='maximum')

        if non_linearities:
            self.relu = BiasRelu([
                (repr_out[0], True),
                (repr_out[1] * 3, False),
                (repr_out[2] * 5, False),
                (repr_out[1] + repr_out[2], True)], normalize=False)
        else:
            self.relu = None

    def forward(self, x): # pylint: disable=W
        y = self.bn_conv(x)

        if self.relu is None:
            return y

        y = self.relu(y)

        if self.repr_out[1] + self.repr_out[2] == 0:
            return y

        nbatch = y.size(0)
        nx = y.size(2)
        ny = y.size(3)
        nz = y.size(4)

        begin_y = self.repr_out[0]
        begin_u = self.repr_out[0] + self.repr_out[1] * 3 + self.repr_out[2] * 5

        zs = [y[:, :self.repr_out[0]]]

        for (m, dim) in [(self.repr_out[1], 3), (self.repr_out[2], 5)]:
            if m == 0:
                continue

            field_y = y[:, begin_y: begin_y + m * dim] # [batch, feature * repr, x, y, z]
            field_y = field_y.contiguous()
            field_y = field_y.view(nbatch, m, dim, nx, ny, nz) # [batch, feature, repr, x, y, z]

            field_u = y[:, begin_u: begin_u + m] # [batch, feature, x, y, z]
            field_u = field_u.contiguous()
            field_u = field_u.view(nbatch, m, 1, nx, ny, nz) # [batch, feature, repr, x, y, z]

            field = field_y * field_u # [batch, feature, repr, x, y, z]
            field = field.view(nbatch, m * dim, nx, ny, nz) # [batch, feature * repr, x, y, z]

            zs.append(field)

            begin_y += m * dim
            begin_u += m

        return torch.cat(zs, dim=1)
