# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import o3, rs


class SO3Activation(torch.nn.Module):
    def __init__(self, Rs, act, n):
        '''
        map to a signal on SO3, apply the non linearity point wise and project back
        the signal on SO3 is the regular representation of SO3
        and we can apply a pointwise operation on these representations

        :param Rs: input representation
        :param act: activation function
        :param n: number of point on the sphere (the higher the more accurate)
        '''
        super().__init__()

        Rs = rs.simplify(Rs)
        mul0, _, _ = Rs[0]
        assert all(mul0 * (2 * l + 1) == mul for mul, l, _ in Rs)
        assert [l for _, l, _ in Rs] == list(range(len(Rs)))
        assert all(p == 0 for _, l, p in Rs)

        self.Rs_out = Rs

        x = [o3.rand_rot() for _ in range(n)]
        Z = torch.stack([torch.cat([o3.irr_repr(l, *o3.rot_to_abc(R)).flatten() * (2 * l + 1)**0.5 for l in range(len(Rs))]) for R in x])  # [z, lmn]
        Z.div_(Z.shape[1]**0.5)
        self.register_buffer('Z', Z)
        self.act = act

    def forward(self, features, dim=-1):
        '''
        :param features: [..., channels, ...]
        expected normalization: component
        '''
        assert features.shape[dim] == self.Z.shape[1]  # assert mul == 1 for now
        features = features.transpose(0, dim)
        out_features = (self.Z.shape[1] / self.Z.shape[0]) * self.Z.T @ self.act(self.Z @ features.flatten(1))
        out_features = out_features.reshape(*features.shape)
        out_features = out_features.transpose(0, dim)
        return out_features
