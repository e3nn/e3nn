# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3BNConvolution, SE3ConvolutionBN
from se3_cnn.non_linearities import ScalarActivation
from se3_cnn import SO3


class GatedActivation(torch.nn.Module):
    def __init__(self,
                 repr_in, size, radial_window_dict,  # kernel params
                 activation=(None, None),  # nonlinearity
                 batch_norm_momentum=0.1, batch_norm_mode='normal', batch_norm_before_conv=True):  # batch norm params
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param int size: the filters are cubes of dimension = size x size x size
        :param radial_window_dict: contains both radial window function and the keyword arguments for the radial window function
        :param activation: (scalar activation, gate activation) which are functions like torch.nn.functional.relu or None
        :param float batch_norm_momentum: batch normalization momentum (put it to zero to disable the batch normalization)
        :param batch_norm_mode: the mode of the batch normalization
        :param bool batch_norm_before_conv: perform the batch normalization before or after the convolution
        '''
        super().__init__()

        if type(activation) is tuple:
            self.scalar_activation, gate_activation = activation
        else:
            self.scalar_activation, gate_activation = activation, activation

        irreducible_repr = [SO3.repr1, SO3.repr3, SO3.repr5, SO3.repr7, SO3.repr9, SO3.repr11, SO3.repr13, SO3.repr15]

        self.repr_in = repr_in
        n_non_scalar = sum(repr_in[1:])

        if gate_activation is not None and n_non_scalar > 0:
            assert size % 2 == 1, "This size needs to be odd such that the gates matches well with the non-scalar fields"
            self.gates = torch.nn.Sequential(
                (SE3BNConvolution if batch_norm_before_conv else SE3ConvolutionBN)(
                    Rs_in=list(zip(repr_in, irreducible_repr)),
                    Rs_out=[(n_non_scalar, SO3.repr1)],
                    size=size,
                    radial_window_dict=radial_window_dict,
                    padding=size // 2,
                    momentum=batch_norm_momentum,
                    mode=batch_norm_mode),
                ScalarActivation([(n_non_scalar, gate_activation)])
            )
        else:
            self.gates = None

    def forward(self, x):  # pylint: disable=W

        # gates
        if self.gates is not None:
            g = self.gates(x)

        nbatch = x.size(0)
        nx = x.size(2)
        ny = x.size(3)
        nz = x.size(4)

        begin_x = 0  # index of first non-scalar capsule
        begin_g = 0  # index of first scalar gate capsule

        zs = []

        for n, mul in enumerate(self.repr_in):
            if mul == 0:
                continue
            dim = 2 * n + 1

            # crop out capsules of order n
            field_x = x[:, begin_x: begin_x + mul * dim]  # [batch, feature * repr, x, y, z]
            begin_x += mul * dim

            if n == 0:
                if self.scalar_activation is not None:
                    field = self.scalar_activation(field_x)
                else:
                    field = field_x
            else:
                if self.gates is not None:
                    # reshape channels in capsules and capsule entries
                    field_x = field_x.contiguous()
                    field_x = field_x.view(nbatch, mul, dim, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                    # crop out corresponding scalar gates
                    field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, x, y, z]
                    begin_g += mul
                    # reshape channels for broadcasting
                    field_g = field_g.contiguous()
                    field_g = field_g.view(nbatch, mul, 1, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                    # scale non-scalar capsules by gate values
                    field = field_x * field_g  # [batch, feature, repr, x, y, z]
                    field = field.view(nbatch, mul * dim, nx, ny, nz)  # [batch, feature * repr, x, y, z]
                else:
                    field = field_x

            zs.append(field)

        return torch.cat(zs, dim=1)  # does not contain gates
