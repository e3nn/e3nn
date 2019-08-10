# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch

from se3cnn.SO3 import normalizeRs


class GatedBlock(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, scalar_activation, gate_activation, Operation, dim=-1):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param scalar_activation: nonlinear function applied on l=0 channels
        :param gate_activation: nonlinear function applied on the gates
        :param Operation: class of signature (Rs_in, Rs_out)
        """
        super().__init__()

        Rs_out = normalizeRs(Rs_out)
        Rs_in = normalizeRs(Rs_in)

        self.Rs_out = Rs_out

        num_scalar = sum(mul for mul, l, p in Rs_out if l == 0)
        if scalar_activation is not None and num_scalar > 0:
            self.scalar_act = scalar_activation
            if any(p == -1 for mul, l, p in Rs_out if l == 0):
                x = torch.linspace(0, 2, 10)
                assert (scalar_activation(-x) + scalar_activation(x)).abs().max() < 1e-5, "need odd function for odd scalars"
        else:
            self.scalar_act = None

        num_non_scalar = sum(mul for mul, l, p in Rs_out if l != 0)
        if gate_activation is not None and num_non_scalar > 0:
            Rs_out_parity = []
            Rs_out_gates = []
            for mul, l, p in Rs_out:
                if p == 0:
                    Rs_out_parity.append((mul, l, 0))
                    Rs_out_gates.append((mul, 0, 0))
                else:
                    Rs_out_parity.append((mul, l, p))
                    Rs_out_gates.append((mul, 0, 1))
                    # Rs_out_parity.append((mul, l, -p))
                    # Rs_out_gates.append((mul, 0, -1))

                    # x = torch.linspace(0, 2, 10)
                    # assert (gate_activation(-x) + gate_activation(x)).abs().max() < 1e-5, "need odd function for odd scalars"

            Rs_out_with_gate = Rs_out_parity + normalizeRs(Rs_out_gates)
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
        features = self.op(*args, **kwargs)  # [..., channel, ...]

        if self.scalar_act is None and self.gate_act is None:
            return features

        dim = (features.dim() + self.dim) % features.dim()
        size_bef = features.size()[:dim]
        size = features.size(dim)
        size_aft = features.size()[dim + 1:]

        size_out = sum(mul * (2 * l + 1) for mul, l, p in self.Rs_out)

        if self.gate_act is not None:
            gates = self.gate_act(features.narrow(dim, size_out, size - size_out))
            begin_g = 0  # index of first scalar gate capsule

        out = features.new_empty(*size_bef, size_out, *size_aft)
        begin_out = 0  # index of first capsule
        begin_in = 0

        for mul, l, _p in self.Rs_out:
            if mul == 0:
                continue

            # crop out capsules of order l
            field_y = features.narrow(dim, begin_in, mul * (2 * l + 1))  # [..., feature * repr, ...]
            begin_in += mul * (2 * l + 1)

            if l == 0:
                # Scalar activation
                if self.scalar_act is not None:
                    field = self.scalar_act(field_y)
                else:
                    field = field_y
            else:
                if self.gate_act is not None:
                    # reshape channels in capsules and capsule entries
                    field_y = field_y.reshape(*size_bef, mul, 2 * l + 1, *size_aft)  # [..., feature, repr, ...]

                    # crop out corresponding scalar gates
                    field_g = gates.narrow(dim, begin_g, mul)  # [..., feature, ...]
                    begin_g += mul

                    # reshape channels for broadcasting
                    field_g = field_g.contiguous().unsqueeze(dim + 1)  # [..., feature, 1, ...]

                    # scale non-scalar capsules by gate values
                    field = (field_y * field_g).view(*size_bef, mul * (2 * l + 1), *size_aft)  # [..., feature * repr, ...]
                    del field_g
                else:
                    field = field_y
            del field_y

            out.narrow(dim, begin_out, mul * (2 * l + 1)).copy_(field)
            begin_out += mul * (2 * l + 1)
            del field

        return out
