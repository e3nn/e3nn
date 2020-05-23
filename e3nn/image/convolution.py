# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, line-too-long, no-member, invalid-name
import math
from functools import partial

import torch

from e3nn import rsh, o3
from e3nn.kernel_mod import FrozenKernel
from e3nn.non_linearities.rescaled_act import swish
from e3nn.radial import CosineBasisModel


class Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, steps=(1, 1, 1), lmax=None, fuzzy_pixels=False, **kwargs):
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
        self.kernel = FrozenKernel(
            Rs_in,
            Rs_out,
            R,
            r,
            selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax),
            normalization='component'
        )
        self.kwargs = kwargs

        if fuzzy_pixels:
            # re-evaluate spherical harmonics by adding randomness
            r = r.reshape(-1, 3)
            r = r[self.kernel.radii > 0]
            rand = torch.rand(20**3, *r.shape).mul(2).sub(1)  # [-1, 1]
            rand.mul_(1 / (size - 1))
            rand[:, :, 0].mul_(steps[0] / min(steps))
            rand[:, :, 1].mul_(steps[1] / min(steps))
            rand[:, :, 2].mul_(steps[2] / min(steps))
            r = rand + r.unsqueeze(0)  # [rand, batch, R^3]
            Y = rsh.spherical_harmonics_xyz([(1, l, p) for _, l, p in self.kernel.Rs_f], r)
            # Y  # [rand, batch, l_filter * m_filter]
            Y.mul_(math.sqrt(4 * math.pi))  # normalization='component'
            self.kernel.Y.copy_(Y.mean(0))

    def forward(self, features):
        """
        :param tensor features: [batch, x, y, z, channel]
        :return: [batch, x, y, z, channel]
        """
        features = torch.einsum('txyzi->tixyz', features)
        k = torch.einsum('xyzij->ijxyz', self.kernel())
        k.mul_(1 / k[0, 0].numel() ** 0.5)
        features = torch.nn.functional.conv3d(features, k, **self.kwargs)
        features = torch.einsum('tixyz->txyzi', features)
        return features
