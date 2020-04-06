# pylint: disable=C,E1101,E1102
import unittest
import math
import torch

from e3nn.linear import Linear
from e3nn import rs


class Tests(unittest.TestCase):
    def test1(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(5, 0), (20, 1), (15, 0), (20, 2)]
        Rs_out = [(5, 0), (10, 1), (10, 2), (5, 0)]

        with torch.no_grad():
            lin = Linear(Rs_in, Rs_out)
            features = torch.randn(10000, rs.dim(Rs_in))
            features = lin(features)

        bins, left, right = 100, -4, 4
        bin_width = (right - left) / (bins - 1)
        x = torch.linspace(left, right, bins)
        p = torch.histc(features, bins, left, right) / features.numel() / bin_width
        q = x.pow(2).div(-2).exp().div(math.sqrt(2 * math.pi))  # Normal law

        # import matplotlib.pyplot as plt
        # plt.plot(x, p)
        # plt.plot(x, q)
        # plt.show()

        Dkl = ((p + 1e-100) / q).log().mul(p).sum()  # Kullback-Leibler divergence of P || Q
        self.assertLess(Dkl, 0.1)

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


if __name__ == '__main__':
    unittest.main()
