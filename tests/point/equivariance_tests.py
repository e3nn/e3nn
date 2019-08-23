# pylint: disable=invalid-name, missing-docstring, no-member, line-too-long
import unittest
from functools import partial

import torch

from se3cnn.non_linearities.gated_block_parity import GatedBlockParity
from se3cnn.non_linearities.rescaled_act import relu, sigmoid, tanh, absolute
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import ConstantRadialModel
from se3cnn.SO3 import (clebsch_gordan, direct_sum, irr_repr, rot,
                        spherical_harmonics_xyz)
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
            r = torch.randn(3)

            abc = torch.randn(3)
            D_in = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_in for _ in range(mul)])
            D_out = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_out for _ in range(mul)])


            W1 = D_out @ k(r)  # [i, j]
            W2 = k(rot(*abc) @ r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())


    def test3(self):
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0, 1), (2, 1, 1), (2, 2, -1)]
            Rs_out = [(2, 0, -1), (2, 1, 1), (2, 2, 1)]

            k = Kernel(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(3)

            D_in = direct_sum(*[p * torch.eye(2 * l + 1) for mul, l, p in Rs_in for _ in range(mul)])
            D_out = direct_sum(*[p * torch.eye(2 * l + 1) for mul, l, p in Rs_out for _ in range(mul)])


            W1 = D_out @ k(r)  # [i, j]
            W2 = k(-r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())


    def test4(self):
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(6) for p in [-1, 1]]

            K = partial(Kernel, RadialModel=ConstantRadialModel)
            C = partial(Convolution, K)

            scalars = [(mul, 0, +1), (mul, 0, -1)], [(mul, relu), (mul, absolute)]
            rs_nonscalars = [(mul, 1, +1), (mul, 1, -1), (mul, 2, +1), (mul, 2, -1), (mul, 3, +1), (mul, 3, -1)]
            n = 3 * mul
            gates = [(n, 0, +1), (n, 0, -1)], [(n, sigmoid), (n, tanh)]

            f = GatedBlockParity(C, Rs_in, *scalars, *gates, rs_nonscalars)

            D_in = direct_sum(*[p * torch.eye(2 * l + 1) for mul, l, p in Rs_in for _ in range(mul)])
            D_out = direct_sum(*[p * torch.eye(2 * l + 1) for mul, l, p in f.Rs_out for _ in range(mul)])

            fea = torch.randn(1, 4, sum(mul * (2 * l + 1) for mul, l, p in Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, f(fea, geo)))
            x2 = f(torch.einsum("ij,zaj->zai", (D_in, fea)), -geo)
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

unittest.main()
