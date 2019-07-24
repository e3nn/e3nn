# pylint: disable=invalid-name, missing-docstring, no-member
import unittest

import torch

from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel
from se3cnn.SO3 import direct_sum, irr_repr, clebsch_gordan, spherical_harmonics_xyz, rot
from se3cnn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test1(self):
        with torch_default_dtype(torch.float64):
            l_in = 3
            l_out = 2

            for l_f in range(abs(l_in - l_out), l_in + l_out + 1):
                r = torch.randn(100, 3)
                Q = clebsch_gordan(l_out, l_in, l_f)

                abc = torch.randn(3)
                D_in = irr_repr(l_in, *abc)
                D_out = irr_repr(l_out, *abc)

                Y = spherical_harmonics_xyz(l_f, r @ rot(*abc).t())
                W = torch.einsum("ijk,kz->zij", (Q, Y))
                W1 = torch.einsum("zij,jk->zik", (W, D_in))

                Y = spherical_harmonics_xyz(l_f, r)
                W = torch.einsum("ijk,kz->zij", (Q, Y))
                W2 = torch.einsum("ij,zjk->zik", (D_out, W))

                self.assertLess((W1 - W2).norm(), 1e-5 * W.norm(), l_f)


    def test2(self):
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0), (0, 1), (2, 2)]
            Rs_out = [(2, 0), (2, 1), (2, 2)]

            k = Kernel(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(100, 3)

            abc = torch.randn(3)
            D_in = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_in for _ in range(mul)])
            D_out = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_out for _ in range(mul)])


            W1 = k(r)  # [batch, i, j]
            W2 = k(r @ rot(*abc).t())  # [batch, i, j]
            W2 = torch.einsum("ij,zjk,kl->zil", (D_out.t(), W2, D_in))
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())

unittest.main()
