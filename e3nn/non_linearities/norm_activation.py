# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long, unbalanced-tuple-unpacking
import torch

from e3nn import rs
from e3nn.non_linearities.norm import Norm


class NormActivation(torch.nn.Module):
    def __init__(self, Rs, activation, normalization='component'):
        super().__init__()

        self.Rs = rs.convention(Rs)
        self.activation = activation
        self.norm = Norm(self.Rs, normalization)
        self.bias = torch.nn.Parameter(torch.zeros(rs.mul_dim(self.Rs)))


    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensor [..., channel]
        """
        *size, d = features.shape
        assert d == rs.dim(self.Rs)

        norms = self.norm(features)  # [..., l*mul]

        output = []
        index_features = 0
        index_norms = 0
        for mul, l, _ in self.Rs:
            v = features.narrow(-1, index_features, mul * (2 * l + 1)).reshape(*size, mul, 2 * l + 1)  # [..., u, i]
            index_features += mul * (2 * l + 1)

            n = norms.narrow(-1, index_norms, mul).reshape(*size, mul, 1)  # [..., u, i]
            b = self.bias[index_norms: index_norms + mul].reshape(mul, 1)  # [u, i]
            index_norms += mul

            if l == 0:
                out = self.activation(v + b)
            else:
                out = self.activation(n + b) * v

            output.append(out.reshape(*size, mul * (2 * l + 1)))

        return torch.cat(output, dim=-1)
