# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch


class GatedBlock(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, scalar_activation, gate_activation, Operation):
        """
        :param Rs_in: input list of (multiplicities, orders)
        :param Rs_out: output list of (multiplicities, orders)
        :param scalar_activation: nonlinear function applied on l=0 channels
        :param gate_activation: nonlinear function applied on the gates
        :param Operation: class of signature (Rs_in, Rs_out)
        """
        super().__init__()

        self.Rs_out = Rs_out

        num_scalar = sum(mul for mul, l in Rs_out if l == 0)
        if scalar_activation is not None and num_scalar > 0:
            self.scalar_act = scalar_activation
        else:
            self.scalar_act = None

        num_non_scalar = sum(mul for mul, l in Rs_out if l != 0)
        if gate_activation is not None and num_non_scalar > 0:
            Rs_out_with_gate = Rs_out + [(num_non_scalar, 0)]
            self.gate_act = gate_activation
        else:
            Rs_out_with_gate = Rs_out
            self.gate_act = None

        self.op = Operation(Rs_in, Rs_out_with_gate)


    def forward(self, *args, **kwargs):
        """
        :return: tensor [batch, channel, ...]
        """
        y = self.op(*args, **kwargs)  # [batch, channel, ...]

        if self.scalar_act is None and self.gate_act is None:
            return y

        batch = y.size(0)
        size = y.size()[2:]

        size_out = sum(mul * (2 * l + 1) for mul, l in self.Rs_out)

        if self.gate_act is not None:
            g = self.gate_act(y[:, size_out:])
            begin_g = 0  # index of first scalar gate capsule

        z = y.new_empty(batch, size_out, *size)
        begin_y = 0  # index of first capsule

        for mul, l in self.Rs_out:
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
