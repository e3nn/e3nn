# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
import unittest

import torch

from e3nn import rs
from e3nn.non_linearities.s2 import S2Activation


class Tests(unittest.TestCase):

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        def test(Rs, act):
            x = torch.randn(2, sum(2 * l + 1 for _, l, _ in Rs))
            ac = S2Activation(Rs, act, 200)

            a, b, c, p = *torch.rand(3), 1
            y1 = ac(x) @ rs.rep(ac.Rs_out, a, b, c, p).T
            y2 = ac(x @ rs.rep(Rs, a, b, c, p).T)
            self.assertLess((y1 - y2).abs().max(), 1e-4 * y1.abs().max())

        lmax = 5
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
