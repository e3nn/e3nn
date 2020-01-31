# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, line-too-long, no-member, invalid-name
from functools import partial

import torch

from e3nn.non_linearities import rescaled_act
from e3nn.kernel import Kernel
from e3nn.radial import CosineBasisModel


class Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, **kwargs):
        super().__init__()

        R = partial(CosineBasisModel, max_radius=1.0, number_of_basis=(size+1)//2, h=50, L=3, act=rescaled_act.relu)
        self.kernel = Kernel(Rs_in, Rs_out, R, normalization='component')
        x = torch.linspace(-1, 1, size)
        self.r = torch.stack(torch.meshgrid(x, x, x), dim=-1)
        self.kwargs = kwargs

    def forward(self, features):
        k = torch.einsum('xyzij->ijxyz', self.kernel(self.r))
        k.mul_(2 / k[0, 0].numel() ** 0.5)
        return torch.nn.functional.conv3d(features, k, **self.kwargs)
