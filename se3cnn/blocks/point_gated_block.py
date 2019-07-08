# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ
import torch

from se3cnn.convolution import SE3PointConvolution
from se3cnn.non_linearities import ScalarActivation


# Added this for tetris_point.py
# TODO: Refactor so we don't need separate functions for point and image
class PointGatedBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, Kernel, activation=(None, None)):
        super().__init__()

        if isinstance(activation, tuple):
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation

        self.repr_out = repr_out

        Rs_in = [(m, l) for l, m in enumerate(repr_in)]
        Rs_out = [(m, l) for l, m in enumerate(repr_out)]

        if scalar_activation is not None and repr_out[0] > 0:
            self.scalar_act = ScalarActivation([(repr_out[0], scalar_activation)], bias=False)
        else:
            self.scalar_act = None

        num_non_scalar = sum(repr_out[1:])
        if gate_activation is not None and num_non_scalar > 0:
            Rs_out_with_gate = Rs_out + [(num_non_scalar, 0)]
            self.gate_act = ScalarActivation([(num_non_scalar, gate_activation)], bias=False)
        else:
            self.gate_act = None

        self.conv = SE3PointConvolution(Kernel(Rs_in, Rs_out_with_gate))


    def forward(self, input, difference_matrix, relative_mask=None):
        y = self.conv(input, difference_matrix, relative_mask)  # [batch, channel, N]

        if self.scalar_act is None and self.gate_act is None:
            return y

        has_batch = difference_matrix.dim() == 4
        if not has_batch:
            difference_matrix = difference_matrix.unsqueeze(0)
            y = y.unsqueeze(0)

        batch, N, _M, _ = difference_matrix.size()

        size_out = sum(mul * (2 * l + 1) for l, mul in enumerate(self.repr_out))

        if self.gate_act is not None:
            g = y[:, size_out:]
            g = self.gate_act(g)
            begin_g = 0  # index of first scalar gate capsule

        z = y.new_empty(batch, size_out, N)
        begin_y = 0  # index of first capsule

        for l, mul in enumerate(self.repr_out):
            if mul == 0:
                continue
            dim = 2 * l + 1

            # crop out capsules of order l
            field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, N]

            if l == 0:
                # Scalar activation
                if self.scalar_act is not None:
                    field = self.scalar_act(field_y)
                else:
                    field = field_y
            else:
                if self.gate_act is not None:
                    # reshape channels in capsules and capsule entries
                    field_y = field_y.contiguous()
                    field_y = field_y.view(batch, mul, dim, N)  # [batch, feature, repr, N]

                    # crop out corresponding scalar gates
                    field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, N]
                    begin_g += mul
                    # reshape channels for broadcasting
                    field_g = field_g.contiguous()
                    field_g = field_g.view(batch, mul, 1, N)  # [batch, feature, 1, N]

                    # scale non-scalar capsules by gate values
                    field = field_y * field_g  # [batch, feature, repr, N]
                    field = field.view(batch, mul * dim, N)  # [batch, feature * repr, N]
                    del field_g
                else:
                    field = field_y
            del field_y

            z[:, begin_y: begin_y + mul * dim] = field
            begin_y += mul * dim
            del field

        if not has_batch:
            z = z.squeeze(0)

        return z
