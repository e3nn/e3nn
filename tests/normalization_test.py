# pylint: disable=invalid-name, missing-docstring, no-member
import unittest
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.radial import ConstantRadialModel
from e3nn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test1(self):
        with torch_default_dtype(torch.float64):
            mul = 100000
            for l_in in range(3 + 1):
                Rs_in = [(mul, l_in)]
                for l_out in range(2 + 1):
                    Rs_out = [(1, l_out)]

                    k = Kernel(Rs_in, Rs_out, ConstantRadialModel, normalization='component',
                               selection_rule=partial(o3.selection_rule_in_out_sh, lmax=3))
                    k = k(torch.randn(1, 3))

                    self.assertLess(k.mean().item(), 1e-3)
                    self.assertAlmostEqual(k.var().item() * rs.dim(Rs_in), 1, places=1)

    def test2(self):
        with torch_default_dtype(torch.float64):
            mul = 100000
            for l_in in range(3 + 1):
                Rs_in = [(mul, l_in)]
                for l_out in range(2 + 1):
                    Rs_out = [(1, l_out)]

                    k = Kernel(Rs_in, Rs_out, ConstantRadialModel, normalization='norm',
                               selection_rule=partial(o3.selection_rule_in_out_sh, lmax=3))
                    k = k(torch.randn(1, 3))

                    self.assertLess(k.mean().item(), 1e-3)
                    self.assertAlmostEqual(k.var().item() * mul, 1 / (2 * l_out + 1), places=1)


if __name__ == '__main__':
    unittest.main()
