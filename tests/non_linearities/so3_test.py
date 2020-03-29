# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import rs
from e3nn.non_linearities.so3 import SO3Activation


class Tests(unittest.TestCase):

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        def test(Rs, ac):
            x = torch.randn(99, rs.dim(Rs))
            a, b = torch.rand(2)
            c = 1

            y1 = ac(x, dim=-1) @ rs.rep(ac.Rs_out, a, b, c).T
            y2 = ac(x @ rs.rep(Rs, a, b, c).T, dim=-1)
            y3 = ac(x @ rs.rep(Rs, -c, -b, -a).T, dim=-1)
            self.assertLess((y1 - y2).norm(), (y1 - y3).norm())

        L = 5
        Rs = [(2 * l + 1, l) for l in range(L + 1)]
        ac = SO3Activation(Rs, torch.abs, 500)
        acts = [torch.tanh, torch.abs, lambda x: x]

        for act in acts:
            ac.act = act
            for _ in range(10):
                test(Rs, ac)


if __name__ == '__main__':
    unittest.main()
