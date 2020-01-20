# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
from functools import reduce

import torch

from e3nn.point.kernel import Kernel
from e3nn.point.radial import ConstantRadialModel
from e3nn import SO3


class SortSphericalSignals(torch.nn.Module):
    """
    Sort the representation order of a tensor
    """
    def __init__(self, Rs):
        super().__init__()

        self.Rs_in = SO3.normalizeRs(Rs)
        xs = []

        j = 0  # input offset
        for mul, l, p in self.Rs_in:
            d = mul * (2 * l + 1)
            xs.append((l, p, mul, j, d))
            j += d

        mixing_matrix = torch.zeros(j, j)

        Rs_out = []
        i = 0  # output offset
        for l, p, mul, j, d in sorted(xs):
            Rs_out.append((mul, l, p))
            mixing_matrix[i:i+d, j:j+d] = torch.eye(d)
            i += d

        self.Rs_out = SO3.normalizeRs(Rs_out)
        self.register_buffer('mixing_matrix', mixing_matrix)

    def forward(self, x):
        """
        :param x: tensor [batch, feature, ...]
        """
        output = torch.einsum('ij,zja->zia', (self.mixing_matrix, x.flatten(2))).contiguous()
        return output.view(*x.size())


class ConcatenateSphericalSignals(torch.nn.Module):
    """
    Concatenate tensors
    """
    def __init__(self, *Rs):
        super().__init__()
        Rs = reduce(list.__add__, Rs, [])
        self.sort = SortSphericalSignals(Rs)
        self.Rs_out = self.sort.Rs_out

    def forward(self, *signals):
        combined = torch.cat(signals, dim=1)
        return self.sort(combined)


class SelfInteraction(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out, ConstantRadialModel)

    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensor [..., channel]
        """
        *size, n = features.size()
        features = features.view(-1, n)

        k = self.kernel(features.new_zeros(features.size(0), 3))
        features = torch.einsum("zij,zj->zi", (k, features))
        features = features.view(*size, -1)
        return features
