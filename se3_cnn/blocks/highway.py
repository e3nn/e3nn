# pylint: disable=C,R,E1101
import torch
from se3_cnn.bn_conv import SE3BNConvolution
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn import SO3

class HighwayBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, non_linearities, stride):
        '''
        :param: repr_in: tuple with multiplicities of repr. (1, 3, 5)
        :param: repr_out: same but for the output
        :param: non_linearities: boolean, enable or not relu
        :param: stride: stride of the convolution
        '''
        super().__init__()
        self.repr_out = repr_out
        self.bn_conv = SE3BNConvolution(
            size=7,
            radial_amount=3,
            Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5)],
            Rs_out=[(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)],
            stride=stride,
            padding=3,
            momentum=0.01,
            mode='maximum')

        self.non_linearities = non_linearities
        if non_linearities:
            self.relu = BiasRelu([(repr_out[0], True), (repr_out[1] * 3, False), (repr_out[2] * 5, False)], normalize=False)
            if repr_out[1] + repr_out[2] > 0:
                self.bn_conv_gate = SE3BNConvolution(
                    size=7,
                    radial_amount=3,
                    Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5)],
                    Rs_out=[(repr_out[1] + repr_out[2], SO3.repr1)],
                    stride=stride,
                    padding=3,
                    momentum=0.01,
                    mode='maximum')
                self.relu_gate = BiasRelu([(repr_out[1] + repr_out[2], True)], normalize=False)
            else:
                self.bn_conv_gate = None
                self.relu_gate = None

    def forward(self, x): # pylint: disable=W
        y = self.bn_conv(x)

        if self.non_linearities:
            y = self.relu(y)

            if self.bn_conv_gate is None:
                return y

            u = self.bn_conv_gate(x)
            u = self.relu_gate(u)

            nbatch = y.size(0)
            nx = y.size(2)
            ny = y.size(3)
            nz = y.size(4)

            zs = [y[:, :self.repr_out[0]]]

            if self.repr_out[1] + self.repr_out[2] > 0:
                begin_y = self.repr_out[0]
                begin_u = 0

                for (m, dim) in [(self.repr_out[1], 3), (self.repr_out[2], 5)]:
                    if m == 0:
                        continue
                    field_y = y[:, begin_y: begin_y + m * dim] # [batch, feature * repr, x, y, z]
                    field_y = field_y.contiguous()
                    field_y = field_y.view(nbatch, m, dim, nx, ny, nz) # [batch, feature, repr, x, y, z]
                    field_u = u[:, begin_u: begin_u + m] # [batch, feature, x, y, z]
                    field_u = field_u.contiguous()
                    field_u = field_u.view(nbatch, m, 1, nx, ny, nz) # [batch, feature, repr, x, y, z]
                    field = field_y * field_u # [batch, feature, repr, x, y, z]
                    field = field.view(nbatch, m * dim, nx, ny, nz) # [batch, feature * repr, x, y, z]
                    zs.append(field)

                    begin_y += m * dim
                    begin_u += m

            z = torch.cat(zs, dim=1)
            return z
        else:
            return y
