# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch


class GatedBlock(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, scalar_activation, gate_activation, Operation, dim=-1):
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
        self.dim = dim


    def forward(self, *args, **kwargs):
        """
        :return: tensor [..., channel, ...]
        """
        y = self.op(*args, **kwargs)  # [..., channel, ...]

        if self.scalar_act is None and self.gate_act is None:
            return y

        dim = (y.dim() + self.dim) % y.dim()
        size_bef = y.size()[:dim]
        size = y.size(dim)
        size_aft = y.size()[dim + 1:]

        size_out = sum(mul * (2 * l + 1) for mul, l in self.Rs_out)

        if self.gate_act is not None:
            g = self.gate_act(y.narrow(dim, size_out, size - size_out))
            begin_g = 0  # index of first scalar gate capsule

        z = y.new_empty(*size_bef, size_out, *size_aft)
        begin_y = 0  # index of first capsule

        for mul, l in self.Rs_out:
            if mul == 0:
                continue

            # crop out capsules of order l
            field_y = y.narrow(dim, begin_y, mul * (2 * l + 1))  # [..., feature * repr, ...]

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
                    field_y = field_y.view(*size_bef, mul, 2 * l + 1, *size_aft)  # [..., feature, repr, ...]

                    # crop out corresponding scalar gates
                    field_g = g.narrow(dim, begin_g, mul)  # [..., feature, ...]
                    begin_g += mul
                    # reshape channels for broadcasting
                    field_g = field_g.contiguous()
                    field_g = field_g.unsqueeze(dim + 1)  # [..., feature, 1, ...]

                    # scale non-scalar capsules by gate values
                    field = field_y * field_g  # [..., feature, repr, ...]
                    field = field.view(*size_bef, mul * (2 * l + 1), *size_aft)  # [..., feature * repr, ...]
                    del field_g
                else:
                    field = field_y
            del field_y

            z.narrow(dim, begin_y, mul * (2 * l + 1)).copy_(field)
            begin_y += mul * (2 * l + 1)
            del field

        return z
