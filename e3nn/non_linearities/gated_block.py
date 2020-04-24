# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long, unbalanced-tuple-unpacking
import torch

from e3nn import rs


class GatedBlock(torch.nn.Module):
    def __init__(self, Rs_out, scalar_activation, gate_activation):
        """
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param scalar_activation: nonlinear function applied on l=0 channels
        :param gate_activation: nonlinear function applied on the gates
        """
        super().__init__()

        Rs_out = rs.simplify(Rs_out)

        self.scalar_act = scalar_activation
        self.gate_act = gate_activation

        Rs = []
        Rs_gates = []
        for mul, l, p in Rs_out:
            if p != 0:
                raise ValueError("use GatedBlockParity instead")
            Rs.append((mul, l))
            if l != 0:
                Rs_gates.append((mul, 0))

        self.Rs = Rs
        self.Rs_in = rs.simplify(Rs + Rs_gates)

    def forward(self, features, dim=-1):
        """
        :param features: tensor [..., channel, ...]
        :return:         tensor [..., channel, ...]
        """
        dim = (features.dim() + dim) % features.dim()
        size_bef = features.size()[:dim]
        size = features.size(dim)
        size_aft = features.size()[dim + 1:]

        size_out = sum(mul * (2 * l + 1) for mul, l in self.Rs)

        gates = features.narrow(dim, size_out, size - size_out)
        begin_g = 0  # index of first scalar gate capsule

        out = features.new_empty(*size_bef, size_out, *size_aft)
        begin_out = 0  # index of first capsule

        for mul, l in self.Rs:
            if mul == 0:
                continue

            # crop out capsules of order l
            sub = features.narrow(dim, begin_out, mul * (2 * l + 1))  # [..., feature * repr, ...]

            if l == 0:
                # Scalar activation
                sub = self.scalar_act(sub)
            else:
                sub = sub.reshape(*size_bef, mul, 2 * l + 1, *size_aft)  # [..., feature, repr, ...]

                gate = gates.narrow(dim, begin_g, mul)  # [..., feature, ...]
                begin_g += mul

                gate = self.gate_act(gate)
                gate = gate.contiguous().unsqueeze(dim + 1)  # [..., feature, 1, ...]

                sub = (sub * gate).reshape(*size_bef, mul * (2 * l + 1), *size_aft)  # [..., feature * repr, ...]
                del gate

            out.narrow(dim, begin_out, mul * (2 * l + 1)).copy_(sub)
            begin_out += mul * (2 * l + 1)
            del sub

        assert begin_g == size - size_out
        assert begin_out == size_out

        return out
