# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
from functools import reduce

import torch

from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel


class SortSphericalSignals(torch.nn.Module):
    def __init__(self, Rs):
        super().__init__()
        ljds = []

        j = 0
        for mul, l in Rs:
            d = mul * (2 * l + 1)
            ljds.append((l, j, d))
            j += d

        mixing_matrix = torch.zeros(j, j)

        i = 0
        for _l, j, d in sorted(ljds):
            mixing_matrix[i:i+d, j:j+d] = torch.eye(d)
            i += d

        self.register_buffer('mixing_matrix', mixing_matrix)

    def forward(self, x):
        """
        :param x: tensor [batch, feature, ...]
        """
        output = torch.einsum('ij,zja->zia', (self.mixing_matrix, x.flatten(2))).contiguous()
        return output.view(*x.size())


class ConcatenateSphericalSignals(torch.nn.Module):
    def __init__(self, *Rs):
        super().__init__()
        Rs = reduce(list.__add__, Rs, [])
        self.sort = SortSphericalSignals(Rs)

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
        :return:         tensro [..., channel]
        """
        *size, n = features.size()
        features = features.view(-1, n)

        k = self.kernel(features.new_zeros(features.size(0), 3))
        features = torch.einsum("zij,zj->zi", (k, features))
        features = features.view(*size, -1)
        return features
