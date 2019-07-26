# pylint: disable=invalid-name, missing-docstring, no-member
import unittest

import torch

from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel
from se3cnn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test1(self):
        with torch_default_dtype(torch.float64):
            mul = 100000
            for l_in in range(4):
                Rs_in = [(mul, l_in)]
                for l_out in range(4):
                    Rs_out = [(1, l_out)]

                    k = Kernel(Rs_in, Rs_out, ConstantRadialModel, normalization='component')
                    k = k(torch.randn(1, 3))

                    self.assertLess(k.mean().item(), 1e-3)
                    self.assertAlmostEqual(k.var().item() * mul * (2 * l_in + 1), 1, places=1)


    def test2(self):
        with torch_default_dtype(torch.float64):
            mul = 100000
            for l_in in range(4):
                Rs_in = [(mul, l_in)]
                for l_out in range(4):
                    Rs_out = [(1, l_out)]

                    k = Kernel(Rs_in, Rs_out, ConstantRadialModel, normalization='norm')
                    k = k(torch.randn(1, 3))

                    self.assertLess(k.mean().item(), 1e-3)
                    self.assertAlmostEqual(k.var().item() * mul, 1 / (2 * l_out + 1), places=1)

unittest.main()
