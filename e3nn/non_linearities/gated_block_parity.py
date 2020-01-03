# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, unbalanced-tuple-unpacking
import torch

from e3nn.non_linearities.activation import Activation
from e3nn.non_linearities.multiplication import Multiplication


def split_features(features, *Rss):
    index = 0
    outputs = []
    for Rs in Rss:
        n = sum(mul * (2 * l + 1) for mul, l, _ in Rs)
        outputs.append(features.narrow(-1, index, n))
        index += n
    assert index == features.size(-1)
    return outputs


class GatedBlockParity(torch.nn.Module):
    def __init__(self, Operation, Rs_in, Rs_scalars, act_scalars, Rs_gates, act_gates, Rs_nonscalars):
        super().__init__()

        self.op = Operation(Rs_in, Rs_scalars + Rs_gates + Rs_nonscalars)
        self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars = Rs_scalars, Rs_gates, Rs_nonscalars

        self.act_scalars = Activation(Rs_scalars, act_scalars)
        Rs_scalars = self.act_scalars.Rs_out

        self.act_gates = Activation(Rs_gates, act_gates)
        Rs_gates = self.act_gates.Rs_out

        self.mul = Multiplication(Rs_nonscalars, Rs_gates)
        Rs_nonscalars = self.mul.Rs_out

        self.Rs_out = Rs_scalars + Rs_nonscalars

    def forward(self, *args, dim=-1, **kwargs):
        features = self.op(*args, **kwargs)
        scalars, gates, nonscalars = split_features(features, self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars)
        scalars = self.act_scalars(scalars)
        if gates.size(dim):
            gates = self.act_gates(gates)
            nonscalars = self.mul(nonscalars, gates)
            features = torch.cat([scalars, nonscalars], dim=dim)
        else:
            features = scalars
        return features
