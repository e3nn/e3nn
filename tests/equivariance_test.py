# pylint: disable=invalid-name, missing-docstring, no-member, line-too-long
import unittest
from functools import partial

import torch

from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.non_linearities.gated_block_parity import GatedBlockParity
from e3nn.non_linearities.rescaled_act import absolute, relu, sigmoid, tanh
from e3nn.kernel_mod import Kernel
from e3nn.point.operations import Convolution
from e3nn.radial import ConstantRadialModel
from e3nn import o3, rs
from e3nn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test1(self):
        """Test irr_repr and clebsch_gordan equivariance."""
        with torch_default_dtype(torch.float64):
            l_in = 3
            l_out = 2

            for l_f in range(abs(l_in - l_out), l_in + l_out + 1):
                r = torch.randn(100, 3)
                Q = o3.clebsch_gordan(l_out, l_in, l_f)

                abc = torch.randn(3)
                D_in = o3.irr_repr(l_in, *abc)
                D_out = o3.irr_repr(l_out, *abc)

                Y = o3.spherical_harmonics_xyz(l_f, r @ o3.rot(*abc).t())
                W = torch.einsum("ijk,kz->zij", (Q, Y))
                W1 = torch.einsum("zij,jk->zik", (W, D_in))

                Y = o3.spherical_harmonics_xyz(l_f, r)
                W = torch.einsum("ijk,kz->zij", (Q, Y))
                W2 = torch.einsum("ij,zjk->zik", (D_out, W))

                self.assertLess((W1 - W2).norm(), 1e-5 * W.norm(), l_f)

    def test2(self):
        """Test rotation equivariance on Kernel."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0), (0, 1), (2, 2)]
            Rs_out = [(2, 0), (2, 1), (2, 2)]

            k = Kernel(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(3)

            abc = torch.randn(3)
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            W1 = D_out @ k(r)  # [i, j]
            W2 = k(o3.rot(*abc) @ r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())

    def test3(self):
        """Test rotation equivariance on GatedBlock and dependencies."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0), (0, 1), (2, 2)]
            Rs_out = [(2, 0), (2, 1), (2, 2)]

            K = partial(Kernel, RadialModel=ConstantRadialModel)

            act = GatedBlock(Rs_out, scalar_activation=sigmoid, gate_activation=sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)

            abc = torch.randn(3)
            rot_geo = o3.rot(*abc)
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            fea = torch.randn(1, 4, sum(mul * (2 * l + 1) for mul, l in Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), torch.einsum("ij,zaj->zai", rot_geo, geo)))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test4(self):
        """Test parity equivariance on Kernel."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0, 1), (2, 1, 1), (2, 2, -1)]
            Rs_out = [(2, 0, -1), (2, 1, 1), (2, 2, 1)]

            k = Kernel(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(3)

            D_in = rs.rep(Rs_in, 0, 0, 0, 1)
            D_out = rs.rep(Rs_out, 0, 0, 0, 1)

            W1 = D_out @ k(r)  # [i, j]
            W2 = k(-r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())

    def test5(self):
        """Test parity equivariance on GatedBlockParity and dependencies."""
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]

            K = partial(Kernel, RadialModel=ConstantRadialModel)

            scalars = [(mul, 0, +1), (mul, 0, -1)], [(mul, relu), (mul, absolute)]
            rs_nonscalars = [(mul, 1, +1), (mul, 1, -1), (mul, 2, +1), (mul, 2, -1), (mul, 3, +1), (mul, 3, -1)]
            n = 3 * mul
            gates = [(n, 0, +1), (n, 0, -1)], [(n, sigmoid), (n, tanh)]

            act = GatedBlockParity(*scalars, *gates, rs_nonscalars)
            conv = Convolution(K, Rs_in, act.Rs_in)

            D_in = rs.rep(Rs_in, 0, 0, 0, 1)
            D_out = rs.rep(act.Rs_out, 0, 0, 0, 1)

            fea = torch.randn(1, 4, sum(mul * (2 * l + 1) for mul, l, p in Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), -geo))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test6(self):
        """Test parity and rotation equivariance on GatedBlockParity and dependencies."""
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]

            K = partial(Kernel, RadialModel=ConstantRadialModel)

            scalars = [(mul, 0, +1), (mul, 0, -1)], [(mul, relu), (mul, absolute)]
            rs_nonscalars = [(mul, 1, +1), (mul, 1, -1), (mul, 2, +1), (mul, 2, -1), (mul, 3, +1), (mul, 3, -1)]
            n = 3 * mul
            gates = [(n, 0, +1), (n, 0, -1)], [(n, sigmoid), (n, tanh)]

            act = GatedBlockParity(*scalars, *gates, rs_nonscalars)
            conv = Convolution(K, Rs_in, act.Rs_in)

            abc = torch.randn(3)
            rot_geo = -o3.rot(*abc)
            D_in = rs.rep(Rs_in, *abc, 1)
            D_out = rs.rep(act.Rs_out, *abc, 1)

            fea = torch.randn(1, 4, sum(mul * (2 * l + 1) for mul, l, p in Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), torch.einsum("ij,zaj->zai", rot_geo, geo)))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())


if __name__ == '__main__':
    unittest.main()
