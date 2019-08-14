# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long, unbalanced-tuple-unpacking
import torch

from se3cnn.non_linearities.rescaled_act import tanh
from se3cnn.SO3 import normalizeRs, haslinearpathRs


def split(n, *ons):
    k = sum(ons)
    i = 0
    out = []
    for on in ons:
        if on:
            if i < n % k:
                out.append(n // k + 1)
            else:
                out.append(n // k)
            i += 1
        else:
            out.append(0)
    return out


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

        self.scalar_act = scalar_activation
        self.gate_act = gate_activation

        has_gate_plus = haslinearpathRs(Rs_in, 0, +1)
        has_gate_minus = haslinearpathRs(Rs_in, 0, -1)

        Rs_parity = []
        Rs_gates = []
        Rs_info = []
        for mul, l, p in Rs_out:
            if l == 0:
                if p == 0:
                    Rs_parity.append((mul, 0, 0))
                    Rs_info.append((mul, l, 0, 0, 0))
                elif p == 1:
                    mul1, mul2 = split(mul, has_gate_plus, has_gate_minus)
                    assert mul == mul1 + mul2

                    Rs_parity.append((mul1, 0, 1))
                    Rs_info.append((mul1, l, 1, 0, 1))

                    Rs_parity.append((mul2, 0, -1))
                    Rs_info.append((mul2, l, -1, 0, 1))
                else:
                    assert has_gate_minus
                    Rs_parity.append((mul, 0, -1))
                    Rs_info.append((mul, l, -1, 0, -1))
            else:
                if p == 0:
                    Rs_parity.append((mul, l, 0))
                    Rs_gates.append((mul, 0, 0))
                    Rs_info.append((mul, l, 0, 0, 0))
                else:
                    has_same = haslinearpathRs(Rs_in, l, p)
                    has_opposit = haslinearpathRs(Rs_in, l, -p)

                    mul1, mul2, mul3 = split(mul, has_same and has_gate_plus, has_opposit and has_gate_minus, has_same and has_gate_minus)
                    assert mul == mul1 + mul2 + mul3

                    Rs_parity.append((mul1, l, p))
                    Rs_gates.append((mul1, 0, 1))
                    Rs_info.append((mul1, l, p, 1, p))

                    Rs_parity.append((mul2, l, -p))
                    Rs_gates.append((mul2, 0, -1))
                    Rs_info.append((mul2, l, -p, -1, p))

                    Rs_parity.append((mul3, l, p))
                    Rs_gates.append((mul3, 0, -1))
                    Rs_info.append((mul3, l, p, -1, p))

        self.Rs_info = Rs_info
        self.op = Operation(Rs_in, normalizeRs(Rs_parity + Rs_gates))
        self.dim = dim


    def forward(self, *args, **kwargs):
        """
        :return: tensor [..., channel, ...]
        """
        features = self.op(*args, **kwargs)  # [..., channel, ...]

        dim = (features.dim() + self.dim) % features.dim()
        size_bef = features.size()[:dim]
        size = features.size(dim)
        size_aft = features.size()[dim + 1:]

        size_out = sum(mul * (2 * l + 1) for mul, l, p_in, p_gate, p_out in self.Rs_info)

        gates = features.narrow(dim, size_out, size - size_out)
        begin_g = 0  # index of first scalar gate capsule

        out = features.new_empty(*size_bef, size_out, *size_aft)
        begin_out = 0  # index of first capsule

        for mul, l, p_in, p_gate, p_out in self.Rs_info:
            if mul == 0:
                continue

            # crop out capsules of order l
            sub = features.narrow(dim, begin_out, mul * (2 * l + 1))  # [..., feature * repr, ...]

            if l == 0:
                # Scalar activation
                if p_in == -1:
                    if p_out == -1:
                        sub = tanh(sub)
                    if p_out == 1:
                        sub = sub.abs()
                else:
                    sub = self.scalar_act(sub)
            else:
                sub = sub.reshape(*size_bef, mul, 2 * l + 1, *size_aft)  # [..., feature, repr, ...]

                gate = gates.narrow(dim, begin_g, mul)  # [..., feature, ...]
                begin_g += mul

                if p_gate == -1:
                    if p_in * p_gate == p_out:
                        gate = tanh(gate)
                    else:
                        gate = gate.abs()
                else:
                    gate = self.gate_act(gate)
                gate = gate.contiguous().unsqueeze(dim + 1)  # [..., feature, 1, ...]

                sub = (sub * gate).view(*size_bef, mul * (2 * l + 1), *size_aft)  # [..., feature * repr, ...]
                del gate

            out.narrow(dim, begin_out, mul * (2 * l + 1)).copy_(sub)
            begin_out += mul * (2 * l + 1)
            del sub

        assert begin_g == size - size_out
        assert begin_out == size_out

        return out
