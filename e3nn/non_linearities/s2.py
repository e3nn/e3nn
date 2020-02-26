# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import math

import torch

from e3nn import SO3


class S2Activation(torch.nn.Module):
    def __init__(self, Rs, act, n):
        '''
        map to the sphere, apply the non linearity point wise and project back
        the signal on the sphere is a quasiregular representation of O3
        and we can apply a pointwise operation on these representations

        :param Rs: input representation
        :param act: activation function
        :param n: number of point on the sphere (the higher the more accurate)
        '''
        super().__init__()

        Rs = SO3.simplifyRs(Rs)
        mul0, _, p0 = Rs[0]
        assert all(mul0 == mul for mul, _, _ in Rs)
        assert [l for _, l, _ in Rs] == list(range(len(Rs)))
        assert all(p == p0 for _, l, p in Rs) or all(p == p0 * (-1) ** l for _, l, p in Rs)

        if p0 == +1 or p0 == 0:
            self.Rs_out = Rs
        if p0 == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = 1
                self.Rs_out = [(mul, l, -p) for mul, l, p in Rs]
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = -1
                self.Rs_out = Rs
            else:
                # p_act = 0
                raise ValueError("warning! the parity is violated")

        x = torch.randn(n, 3)
        x = torch.cat([x, -x])
        Y = SO3.spherical_harmonics_xyz(list(range(len(Rs))), x)  # [lm, z]
        self.register_buffer('Y', Y)
        self.act = act

    def forward(self, features, dim=-1):
        '''
        :param features: [..., channels, ...]
        '''
        assert features.shape[dim] == self.Y.shape[0]  # assert mul == 1 for now
        features = features.transpose(0, dim)
        out_features = (4 * math.pi / self.Y.shape[1]) * self.Y @ self.act(self.Y.T @ features.flatten(1))
        out_features = out_features.view(*features.shape)
        out_features = out_features.transpose(0, dim)
        return out_features
