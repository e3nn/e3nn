# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import rs, soft


class S2Activation(torch.nn.Module):
    def __init__(self, Rs, act, res):
        '''
        map to the sphere, apply the non linearity point wise and project back
        the signal on the sphere is a quasiregular representation of O3
        and we can apply a pointwise operation on these representations

        :param Rs: input representation
        :param act: activation function
        :param res: resolution of the SOFT grid on the sphere (the higher the more accurate)
        '''
        super().__init__()

        Rs = rs.simplify(Rs)
        _, _, p0 = Rs[0]
        mul, lmax, _ = Rs[-1]
        assert all(mul == mulx for mulx, _, _ in Rs)
        assert [l for _, l, _ in Rs] == [l for l in range(lmax + 1)]
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

        self.to_soft = soft.ToSOFT(mul, lmax, res)
        self.from_soft = soft.FromSOFT(mul, res, lmax)
        self.act = act

    def forward(self, features):
        '''
        :param features: [..., l * mul * m]
        '''
        features = self.to_soft(features)  # [..., mul, beta, alpha]
        features = self.act(features)
        features = self.from_soft(features)  # [..., l * mul * m]
        return features
