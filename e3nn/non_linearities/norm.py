# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import rs


class Norm(torch.nn.Module):
    def __init__(self, Rs, normalization='component'):
        super().__init__()

        Rs = rs.simplify(Rs)
        n = sum(mul for mul, _, _ in Rs)
        self.Rs_in = Rs
        self.Rs_out = [(n, 0, +1)]
        self.normalization = normalization

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        output = []
        index = 0
        for mul, l, _ in self.Rs_in:
            sub = features.narrow(-1, index, mul * (2 * l + 1)).reshape(*size, mul, 2 * l + 1)  # [..., u, i]
            index += mul * (2 * l + 1)

            sub = sub.norm(2, dim=-1)  # [..., u]

            if self.normalization == 'component':
                sub = sub / (2 * l + 1) ** 0.5

            output.append(sub)
        assert index == n

        return torch.cat(output, dim=-1)
