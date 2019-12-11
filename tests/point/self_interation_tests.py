# pylint: disable=C,E1101,E1102
import unittest

import torch

from e3nn.point.self_interaction import SelfInteraction


class Tests(unittest.TestCase):
    def test1(self):
        Rs_in = [(2, 0), (0, 1), (2, 2)]
        Rs_out = [(2, 0), (2, 1), (2, 2)]
        m = SelfInteraction(Rs_in, Rs_out)
        n = sum(mul * (2 * l + 1) for mul, l in Rs_in)
        x = torch.randn(2, 4, n)
        y = m(x)
        self.assertTrue(y[:, :, 2: 2+2*3].norm() == 0)

unittest.main()
