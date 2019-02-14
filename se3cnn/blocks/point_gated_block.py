# pylint: disable=no-member
import sys, os
import torch
import se3cnn
import numpy as np

from se3cnn.SO3 import torch_default_dtype
from se3cnn.convolution import SE3PointConvolution
from se3cnn.non_linearities import ScalarActivation

# Added this for tetris_point.py
# TODO: Refactor so we don't need separate functions for point and image
# TODO: Broaden scope of function, add radial function, J_filter_max
class PointGatedBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, radii, activation=(None, None)):
        super().__init__()

        if type(activation) is tuple:
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation

        self.repr_out = repr_out

        Rs_in = [(m, l) for l, m in enumerate(repr_in)]
        Rs_out_with_gate = [(m, l) for l, m in enumerate(repr_out)]

        if scalar_activation is not None and repr_out[0] > 0:
            self.scalar_act = ScalarActivation([(repr_out[0], scalar_activation)], bias=False)
        else:
            self.scalar_act = None

        num_non_scalar = sum(repr_out[1:])
        if gate_activation is not None and num_non_scalar > 0:
            Rs_out_with_gate.append((num_non_scalar, 0))
            self.gate_act = ScalarActivation([(num_non_scalar, gate_activation)], bias=False)
        else:
            self.gate_act = None

        with torch_default_dtype(torch.float64):
            self.conv = SE3PointConvolution(Rs_in, Rs_out_with_gate, radii=radii)

    def forward(self, input, difference_mat):
        y = self.conv(input, difference_mat)
        if self.scalar_act is None and self.gate_act is None:
            z = y
        else:
            if len(difference_mat.size()) == 4:
                batch, N, _M, _ = difference_mat.size()
            if len(difference_mat.size()) == 3:
                N, _M, _ = difference_mat.size()

            size_out = sum(mul * (2 * n + 1) for n, mul in enumerate(self.repr_out))

            if self.gate_act is not None:
                g = y[:, size_out:]
                g = self.gate_act(g)
                begin_g = 0  # index of first scalar gate capsule

            z = y.new_empty((y.size(0), size_out, y.size(2)))
            begin_y = 0  # index of first capsule

            for n, mul in enumerate(self.repr_out):
                if mul == 0:
                    continue
                dim = 2 * n + 1

                # crop out capsules of order n
                field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, N]

                if n == 0:
                    # Scalar activation
                    if self.scalar_act is not None:
                        field = self.scalar_act(field_y)
                    else:
                        field = field_y
                else:
                    if self.gate_act is not None:
                        # reshape channels in capsules and capsule entries
                        field_y = field_y.contiguous()
                        field_y = field_y.view(batch, mul, dim, N)  # [batch, feature, repr, x, y, z]

                        # crop out corresponding scalar gates
                        field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, x, y, z]
                        begin_g += mul
                        # reshape channels for broadcasting
                        field_g = field_g.contiguous()
                        field_g = field_g.view(batch, mul, 1, N)  # [batch, feature, repr, x, y, z]

                        # scale non-scalar capsules by gate values
                        field = field_y * field_g  # [batch, feature, repr, x, y, z]
                        field = field.view(batch, mul * dim, N)  # [batch, feature * repr, x, y, z]
                        del field_g
                    else:
                        field = field_y
                del field_y

                z[:, begin_y: begin_y + mul * dim] = field
                begin_y += mul * dim
                del field
        return z
