# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import rs, soft


class S2Activation(torch.nn.Module):
    def __init__(self, Rs, act, res, normalization='component', lmax_out=None):
        '''
        map to the sphere, apply the non linearity point wise and project back
        the signal on the sphere is a quasiregular representation of O3
        and we can apply a pointwise operation on these representations

        :param Rs: input representation of the form [(1, l, p0 * u^l) for l in [0, ..., lmax]]
        :param act: activation function
        :param res: resolution of the SOFT grid on the sphere (the higher the more accurate)
        :param normalization: either 'norm' or 'component'
        :param lmax_out: maximum l of the output
        '''
        super().__init__()

        Rs = rs.simplify(Rs)
        _, _, p0 = Rs[0]
        _, lmax, _ = Rs[-1]
        assert all(mul == 1 for mul, _, _ in Rs)
        assert [l for _, l, _ in Rs] == [l for l in range(lmax + 1)]
        if all(p == p0 for _, l, p in Rs):
            u = 1
        elif all(p == p0 * (-1) ** l for _, l, p in Rs):
            u = -1
        else:
            assert False, "the parity of the input is not well defined"
        # the input transforms as : A_l ---> p0 * u^l * A_l
        # the sphere signal transforms as : f(r) ---> p0 * f(u * r)
        if lmax_out is None:
            lmax_out = lmax

        if p0 == +1 or p0 == 0:
            self.Rs_out = [(1, l, p0 * u ** l) for l in range(lmax_out + 1)]
        if p0 == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = 1
                self.Rs_out = [(1, l, u ** l) for l in range(lmax_out + 1)]
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = -1
                self.Rs_out = [(1, l, -u ** l) for l in range(lmax_out + 1)]
            else:
                # p_act = 0
                raise ValueError("warning! the parity is violated")

        self.to_soft = soft.ToSOFT(lmax, res, normalization=normalization)
        self.from_soft = soft.FromSOFT(res, lmax_out, normalization=normalization, lmax_in=lmax)
        self.act = act

    def forward(self, features):
        '''
        :param features: [..., l * m]
        '''
        features = self.to_soft(features)  # [..., beta, alpha]
        features = self.act(features)
        features = self.from_soft(features)  # [..., l * m]
        return features
