# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch

from se3cnn.non_linearities import ScalarActivation


class GatedBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, scalar_activation, gate_activation, Convolution):
        """
        :param repr_in: input multiplicities
        :param repr_out: output multiplicities
        :param scalar_activation: nonlinear function applied on l=0 channels
        :param gate_activation: nonlinear function applied on the gates
        :param Convolution: class of signature (Rs_in, Rs_out)
        """
        super().__init__()

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
            Rs_out_with_gate = Rs_out
            self.gate_act = None

        self.conv = Convolution(Rs_in, Rs_out_with_gate)


    def forward(self, *args, **kwargs):
        """
        :return: tensor (batch, channel, ...)
        """
        y = self.conv(*args, **kwargs)  # [batch, channel, ...]

        if self.scalar_act is None and self.gate_act is None:
            return y

        batch = y.size(0)
        size = y.size()[2:]

        size_out = sum(mul * (2 * l + 1) for l, mul in enumerate(self.repr_out))

        if self.gate_act is not None:
            g = y[:, size_out:]
            assert g.size(1) == sum(self.repr_out[1:])
            g = self.gate_act(g)
            begin_g = 0  # index of first scalar gate capsule

        z = y.new_empty(batch, size_out, *size)
        begin_y = 0  # index of first capsule

        for l, mul in enumerate(self.repr_out):
            if mul == 0:
                continue
            dim = 2 * l + 1

            # crop out capsules of order l
            field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, ...]

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
                    field_y = field_y.view(batch, mul, dim, *size)  # [batch, feature, repr, ...]

                    # crop out corresponding scalar gates
                    field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, ...]
                    begin_g += mul
                    # reshape channels for broadcasting
                    field_g = field_g.contiguous()
                    field_g = field_g.view(batch, mul, 1, *size)  # [batch, feature, 1, ...]

                    # scale non-scalar capsules by gate values
                    field = field_y * field_g  # [batch, feature, repr, ...]
                    field = field.view(batch, mul * dim, *size)  # [batch, feature * repr, ...]
                    del field_g
                else:
                    field = field_y
            del field_y

            z[:, begin_y: begin_y + mul * dim] = field
            begin_y += mul * dim
            del field

        return z
