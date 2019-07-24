# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
from functools import reduce

import torch

from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel
from se3cnn.point.utils import convolve


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
        if features.dim() == 2:
            # No batch dimension
            _, N = features.size()
            batch = 1
        if features.dim() == 3:
            # Batch dimension
            batch, _, N = features.size()
        if features.dim() == 4:
            # Multiple atom indices
            batch, _, Na, Nb = features.size()
            N = Na * Nb

        neighbors = torch.arange(N, device=features.device).view(1, N, 1).expand(batch, N, 1)
        output = convolve(self.kernel, features.view(batch, -1, N), features.new_zeros(batch, N, 3), neighbors)

        if features.dim() == 2:
            return output.view(-1, N)
        if features.dim() == 3:
            return output
        if features.dim() == 4:
            return output.view(batch, -1, Na, Nb)
