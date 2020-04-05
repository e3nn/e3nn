# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
import unittest

import torch

from e3nn import o3, rs
from e3nn.non_linearities.s2 import S2Activation
from e3nn.non_linearities.rescaled_act import swish, tanh, sigmoid, softplus, identity, quadratic


class Tests(unittest.TestCase):

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        Rs = [(1, l, (-1) ** l) for l in range(4 + 1)]

        def test(act, normalization):
            x = rs.randn(2, Rs, normalization=normalization)
            ac = S2Activation(Rs, act, 120, normalization=normalization, lmax_out=6, random_rot=True)

            a, b, c = o3.rand_angles()
            y1 = ac(x) @ rs.rep(ac.Rs_out, a, b, c, 1).T
            y2 = ac(x @ rs.rep(Rs, a, b, c, 1).T)
            self.assertLess((y1 - y2).abs().max(), 1e-10 * y1.abs().max())

        acts = [tanh, swish, sigmoid, softplus, identity, quadratic]

        for act, normalization in itertools.product(acts, ['norm', 'component']):
            test(act, normalization)

    def test_equivariance_parity(self):
        torch.set_default_dtype(torch.float64)

        lmax = 5

        def test(Rs, act):
            x = rs.randn(2, Rs)
            ac = S2Activation(Rs, act, 200, lmax_out=lmax + 1, random_rot=True)

            a, b, c, p = *torch.rand(3), 1
            y1 = ac(x) @ rs.rep(ac.Rs_out, a, b, c, p).T
            y2 = ac(x @ rs.rep(Rs, a, b, c, p).T)
            self.assertLess((y1 - y2).abs().max(), 3e-4 * y1.abs().max())

        Rss = [
            [(1, l, -(-1) ** l) for l in range(lmax + 1)],
            [(1, l, (-1) ** l) for l in range(lmax + 1)],
            [(1, l, -1) for l in range(lmax + 1)],
            [(1, l, 1) for l in range(lmax + 1)],
        ]

        acts = [torch.tanh, torch.abs]

        for Rs, act in itertools.product(Rss, acts):
            test(Rs, act)

        Rss = [
            [(1, l, (-1) ** l) for l in range(lmax + 1)],
            [(1, l, 1) for l in range(lmax + 1)],
        ]

        acts = [torch.relu, torch.sigmoid]

        for Rs, act in itertools.product(Rss, acts):
            test(Rs, act)


if __name__ == '__main__':
    unittest.main()
