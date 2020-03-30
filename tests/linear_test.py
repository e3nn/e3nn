# pylint: disable=C,E1101,E1102
import unittest

import torch

from e3nn.linear import Linear
from e3nn.linear_mod import Linear as LinearMod
from e3nn import rs


class Tests(unittest.TestCase):
    def test1(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(50, 0), (150, 1), (50, 0), (100, 2)]
        Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        torch.manual_seed(0)
        lin = Linear(Rs_in, Rs_out)
        features = torch.randn(100, rs.dim(Rs_in))
        features = lin(features)

        self.assertLess(features.pow(2).mean(0).sub(1).abs().max(), 0.4)
    
    def test1_mod(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(50, 0), (150, 1), (50, 0), (100, 2)]
        Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        torch.manual_seed(0)
        lin = LinearMod(Rs_in, Rs_out)
        features = torch.randn(100, rs.dim(Rs_in))
        features = lin(features)

        self.assertLess(features.pow(2).mean(0).sub(1).abs().max(), 0.4)

    def test_equiv(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(5, 0), (15, 1), (5, 0), (10, 2)]
        Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        lin = Linear(Rs_in, Rs_out)
        f_in = torch.randn(100, rs.dim(Rs_in))

        angles = torch.randn(3)
        y1 = lin(torch.einsum('ij,zj->zi', rs.rep(Rs_in, *angles), f_in))
        y2 = torch.einsum('ij,zj->zi', rs.rep(Rs_out, *angles), lin(f_in))

        self.assertLess((y1 - y2).abs().max(), 1e-10)

    def test_equiv_mod(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(5, 0), (15, 1), (5, 0), (10, 2)]
        Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        lin = LinearMod(Rs_in, Rs_out)
        f_in = torch.randn(100, rs.dim(Rs_in))

        angles = torch.randn(3)
        y1 = lin(torch.einsum('ij,zj->zi', rs.rep(Rs_in, *angles), f_in))
        y2 = torch.einsum('ij,zj->zi', rs.rep(Rs_out, *angles), lin(f_in))

        self.assertLess((y1 - y2).abs().max(), 1e-10)


if __name__ == '__main__':
    unittest.main()
