# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import o3, rs, s2grid


class S2Activation(torch.nn.Module):
    def __init__(self, Rs, act, res, normalization='component', lmax_out=None, random_rot=False):
        '''
        map to the sphere, apply the non linearity point wise and project back
        the signal on the sphere is a quasiregular representation of O3
        and we can apply a pointwise operation on these representations

        :param Rs: input representation of the form [(1, l, p_val * (p_arg)^l) for l in [0, ..., lmax]]
        :param act: activation function
        :param res: resolution of the grid on the sphere (the higher the more accurate)
        :param normalization: either 'norm' or 'component'
        :param lmax_out: maximum l of the output
        :param random_rot: rotate randomly the grid
        '''
        super().__init__()

        Rs = rs.simplify(Rs)
        _, _, p_val = Rs[0]
        _, lmax, _ = Rs[-1]
        assert all(mul == 1 for mul, _, _ in Rs)
        assert [l for _, l, _ in Rs] == [l for l in range(lmax + 1)]
        if all(p == p_val for _, l, p in Rs):
            p_arg = 1
        elif all(p == p_val * (-1) ** l for _, l, p in Rs):
            p_arg = -1
        else:
            assert False, "the parity of the input is not well defined"
        self.Rs_in = Rs
        # the input transforms as : A_l ---> p_val * (p_arg)^l * A_l
        # the sphere signal transforms as : f(r) ---> p_val * f(p_arg * r)
        if lmax_out is None:
            lmax_out = lmax

        if p_val == +1 or p_val == 0:
            self.Rs_out = [(1, l, p_val * p_arg ** l) for l in range(lmax_out + 1)]
        if p_val == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = 1
                self.Rs_out = [(1, l, p_arg ** l) for l in range(lmax_out + 1)]
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = -1
                self.Rs_out = [(1, l, -p_arg ** l) for l in range(lmax_out + 1)]
            else:
                # p_act = 0
                raise ValueError("warning! the parity is violated")

        self.to_s2 = s2grid.ToS2Grid(lmax, res, normalization=normalization)
        self.from_s2 = s2grid.FromS2Grid(res, lmax_out, normalization=normalization, lmax_in=lmax)
        self.act = act
        self.random_rot = random_rot

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., l * m]
        '''
        assert features.shape[-1] == rs.dim(self.Rs_in)

        if self.random_rot:
            abc = o3.rand_angles()
            features = torch.einsum('ij,...j->...i', rs.rep(self.Rs_in, *abc), features)

        features = self.to_s2(features)  # [..., beta, alpha]
        features = self.act(features)
        features = self.from_s2(features)

        if self.random_rot:
            features = torch.einsum('ij,...j->...i', rs.rep(self.Rs_out, *abc).T, features)
        return features
