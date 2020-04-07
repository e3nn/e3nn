# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, line-too-long, no-member, invalid-name
from functools import partial

import torch

from e3nn.non_linearities.rescaled_act import swish
from e3nn.kernel_mod import FrozenKernel
from e3nn.radial import CosineBasisModel


class Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, steps=(1, 1, 1), **kwargs):
        super().__init__()

        r = torch.linspace(-1, 1, size)
        x = r * steps[0] / min(steps)
        x = x[x.abs() <= 1]
        y = r * steps[1] / min(steps)
        y = y[y.abs() <= 1]
        z = r * steps[2] / min(steps)
        z = z[z.abs() <= 1]
        r = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]

        R = partial(CosineBasisModel, max_radius=1.0, number_of_basis=(size + 1) // 2, h=50, L=3, act=swish)
        self.kernel = FrozenKernel(Rs_in, Rs_out, R, r, normalization='component')
        self.kwargs = kwargs

    def forward(self, features):
        """
        :param tensor features: [batch, j, x, y, z]
        """
        k = torch.einsum('xyzij->ijxyz', self.kernel())
        k.mul_(1 / k[0, 0].numel() ** 0.5)
        return torch.nn.functional.conv3d(features, k, **self.kwargs)
