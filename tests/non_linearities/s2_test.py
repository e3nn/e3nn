# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
import unittest

import torch

from e3nn import rs
from e3nn.non_linearities.s2 import S2Activation


class Tests(unittest.TestCase):

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        Rs = [(1, l) for l in range(4 + 1)]

        def test(act, normalization):
            x = rs.randn(2, Rs, normalization=normalization)
            ac = S2Activation(Rs, act, 200, normalization=normalization, lmax_out=6)

            a, b, c = torch.rand(3)
            y1 = ac(x) @ rs.rep(ac.Rs_out, a, b, c).T
            y2 = ac(x @ rs.rep(Rs, a, b, c).T)
            self.assertLess((y1 - y2).abs().max(), 3e-4 * y1.abs().max())

        acts = [torch.tanh, torch.abs, torch.relu, torch.sigmoid]

        for act, normalization in itertools.product(acts, ['norm', 'component']):
            test(act, normalization)

    def test_equivariance_parity(self):
        torch.set_default_dtype(torch.float64)

        lmax = 5

        def test(Rs, act):
            x = rs.randn(2, Rs)
            ac = S2Activation(Rs, act, 200, lmax_out=lmax + 1)

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
