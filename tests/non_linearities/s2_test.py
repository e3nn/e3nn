# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
import unittest

import torch

from e3nn import SO3
from e3nn.non_linearities.s2 import S2Activation


class Tests(unittest.TestCase):

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        def test(Rs, act):
            x = torch.randn(55, sum(2 * l + 1 for _, l, _ in Rs))
            ac = S2Activation(Rs, act, 1000)

            y1 = ac(x, dim=-1) @ SO3.rep(ac.Rs_out, 0, 0, 0, -1).T
            y2 = ac(x @ SO3.rep(Rs, 0, 0, 0, -1).T, dim=-1)
            self.assertLess((y1 - y2).abs().max(), 1e-10)

        L = 5
        Rss = [
            [(1, l, -(-1) ** l) for l in range(L)],
            [(1, l, (-1) ** l) for l in range(L)],
            [(1, l, -1) for l in range(L)],
            [(1, l, 1) for l in range(L)],
        ]

        acts = [torch.tanh, torch.abs]

        for Rs, act in itertools.product(Rss, acts):
            test(Rs, act)

        Rss = [
            [(1, l, (-1) ** l) for l in range(L)],
            [(1, l, 1) for l in range(L)],
        ]

        acts = [torch.relu, torch.sigmoid]

        for Rs, act in itertools.product(Rss, acts):
            test(Rs, act)


if __name__ == '__main__':
    unittest.main()
