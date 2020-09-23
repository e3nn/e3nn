# pylint: disable=invalid-name, missing-docstring, no-member, line-too-long
import unittest
from functools import partial

import torch

from e3nn import o3, rs, rsh
from e3nn.kernel import Kernel as Kernel1
from e3nn.kernel_mod import Kernel as KernelMod
from e3nn.linear import Linear as Linear1
from e3nn.linear_mod import Linear as LinearMod
from e3nn.networks import GatedConvNetwork, S2Network
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.non_linearities.gated_block_parity import GatedBlockParity
from e3nn.non_linearities.rescaled_act import absolute, relu, sigmoid, tanh
from e3nn.point.operations import Convolution
from e3nn.radial import ConstantRadialModel
from e3nn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test_irr_repr_wigner_3j(self):
        """Test irr_repr and wigner_3j equivariance."""
        with torch_default_dtype(torch.float64):
            l_in = 3
            l_out = 2

            for l_f in range(abs(l_in - l_out), l_in + l_out + 1):
                r = torch.randn(100, 3)
                Q = o3.wigner_3j(l_out, l_in, l_f)

                abc = torch.randn(3)
                D_in = o3.irr_repr(l_in, *abc)
                D_out = o3.irr_repr(l_out, *abc)

                Y = rsh.spherical_harmonics_xyz([l_f], r @ o3.rot(*abc).t())
                W = torch.einsum("ijk,zk->zij", (Q, Y))
                W1 = torch.einsum("zij,jk->zik", (W, D_in))

                Y = rsh.spherical_harmonics_xyz([l_f], r)
                W = torch.einsum("ijk,zk->zij", (Q, Y))
                W2 = torch.einsum("ij,zjk->zik", (D_out, W))

                self.assertLess((W1 - W2).norm(), 1e-5 * W.norm(), l_f)

    def rotation_kernel(self, K):
        """Test rotation equivariance on Kernel."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0), (0, 1), (2, 2)]
            Rs_out = [(2, 0), (2, 1), (2, 2)]

            k = K(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(3)

            abc = torch.randn(3)
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            W1 = D_out @ k(r)  # [i, j]
            W2 = k(o3.rot(*abc) @ r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())

    def test_rotation_kernel(self):
        self.rotation_kernel(Kernel1)

    def test_rotation_kernel_mod(self):
        self.rotation_kernel(KernelMod)

    def rotation_gated_block(self, K):
        """Test rotation equivariance on GatedBlock and dependencies."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0), (0, 1), (2, 2)]
            Rs_out = [(2, 0), (2, 1), (2, 2)]

            K = partial(K, RadialModel=ConstantRadialModel)

            act = GatedBlock(Rs_out, scalar_activation=sigmoid, gate_activation=sigmoid)
            conv = Convolution(K(Rs_in, act.Rs_in))

            abc = torch.randn(3)
            rot_geo = o3.rot(*abc)
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            fea = torch.randn(1, 4, rs.dim(Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), torch.einsum("ij,zaj->zai", rot_geo, geo)))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test_rotation_gated_block(self):
        self.rotation_gated_block(Kernel1)

    def test_rotation_gated_block_mod(self):
        self.rotation_gated_block(KernelMod)

    def parity_kernel(self, K):
        """Test parity equivariance on Kernel."""
        with torch_default_dtype(torch.float64):
            Rs_in = [(2, 0, 1), (2, 1, 1), (2, 2, -1)]
            Rs_out = [(2, 0, -1), (2, 1, 1), (2, 2, 1)]

            k = K(Rs_in, Rs_out, ConstantRadialModel)
            r = torch.randn(3)

            D_in = rs.rep(Rs_in, 0, 0, 0, 1)
            D_out = rs.rep(Rs_out, 0, 0, 0, 1)

            W1 = D_out @ k(r)  # [i, j]
            W2 = k(-r) @ D_in  # [i, j]
            self.assertLess((W1 - W2).norm(), 10e-5 * W1.norm())

    def test_parity_kernel(self):
        self.parity_kernel(Kernel1)

    def test_parity_kernel_mod(self):
        self.parity_kernel(KernelMod)

    def parity_gated_block_parity(self, K):
        """Test parity equivariance on GatedBlockParity and dependencies."""
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]

            K = partial(K, RadialModel=ConstantRadialModel)

            scalars = [(mul, 0, +1), (mul, 0, -1)], [(mul, relu), (mul, absolute)]
            rs_nonscalars = [(mul, 1, +1), (mul, 1, -1), (mul, 2, +1), (mul, 2, -1), (mul, 3, +1), (mul, 3, -1)]
            n = 3 * mul
            gates = [(n, 0, +1), (n, 0, -1)], [(n, sigmoid), (n, tanh)]

            act = GatedBlockParity(*scalars, *gates, rs_nonscalars)
            conv = Convolution(K(Rs_in, act.Rs_in))

            D_in = rs.rep(Rs_in, 0, 0, 0, 1)
            D_out = rs.rep(act.Rs_out, 0, 0, 0, 1)

            fea = rs.randn(1, 3, Rs_in)
            geo = torch.randn(1, 3, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), -geo))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test_parity_gated_block_parity(self):
        self.parity_gated_block_parity(Kernel1)

    def test_parity_gated_block_parity_mod(self):
        self.parity_gated_block_parity(KernelMod)

    def parity_rotation_gated_block_parity(self, K):
        """Test parity and rotation equivariance on GatedBlockParity and dependencies."""
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]

            K = partial(K, RadialModel=ConstantRadialModel)

            scalars = [(mul, 0, +1), (mul, 0, -1)], [(mul, relu), (mul, absolute)]
            rs_nonscalars = [(mul, 1, +1), (mul, 1, -1), (mul, 2, +1), (mul, 2, -1), (mul, 3, +1), (mul, 3, -1)]
            n = 3 * mul
            gates = [(n, 0, +1), (n, 0, -1)], [(n, sigmoid), (n, tanh)]

            act = GatedBlockParity(*scalars, *gates, rs_nonscalars)
            conv = Convolution(K(Rs_in, act.Rs_in))

            abc = torch.randn(3)
            rot_geo = -o3.rot(*abc)
            D_in = rs.rep(Rs_in, *abc, 1)
            D_out = rs.rep(act.Rs_out, *abc, 1)

            fea = torch.randn(1, 4, rs.dim(Rs_in))
            geo = torch.randn(1, 4, 3)

            x1 = torch.einsum("ij,zaj->zai", (D_out, act(conv(fea, geo))))
            x2 = act(conv(torch.einsum("ij,zaj->zai", (D_in, fea)), torch.einsum("ij,zaj->zai", rot_geo, geo)))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test_parity_rotation_gated_block_parity(self):
        self.parity_rotation_gated_block_parity(Kernel1)

    def test_parity_rotation_gated_block_parity_mod(self):
        self.parity_rotation_gated_block_parity(KernelMod)

    def parity_rotation_linear(self, L):
        """Test parity and rotation equivariance on Linear."""
        with torch_default_dtype(torch.float64):
            mul = 2
            Rs_in = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]
            Rs_out = [(mul, l, p) for l in range(3 + 1) for p in [-1, 1]]

            lin = L(Rs_in, Rs_out)

            abc = torch.randn(3)
            D_in = rs.rep(lin.Rs_in, *abc, 1)
            D_out = rs.rep(lin.Rs_out, *abc, 1)

            fea = torch.randn(rs.dim(Rs_in))

            x1 = torch.einsum("ij,j->i", D_out, lin(fea))
            x2 = lin(torch.einsum("ij,j->i", D_in, fea))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test_parity_rotation_linear(self):
        self.parity_rotation_linear(Linear1)

    def test_parity_rotation_linear_mod(self):
        self.parity_rotation_linear(LinearMod)

    def test_equivariance_gatedconvnetwork(self):
        with torch_default_dtype(torch.float64):
            mul = 3
            Rs_in = [(mul, l) for l in range(3 + 1)]
            Rs_out = [(mul, l) for l in range(3 + 1)]

            net = GatedConvNetwork(Rs_in, [(10, 0), (1, 1), (1, 2), (1, 3)], Rs_out, lmax=3)

            abc = torch.randn(3)
            rot_geo = o3.rot(*abc)
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            fea = torch.randn(1, 10, rs.dim(Rs_in))
            geo = torch.randn(1, 10, 3)

            x1 = torch.einsum("ij,zaj->zai", D_out, net(fea, geo))
            x2 = net(torch.einsum("ij,zaj->zai", D_in, fea), torch.einsum("ij,zaj->zai", rot_geo, geo))
            self.assertLess((x1 - x2).norm(), 10e-5 * x1.norm())

    def test_equivariance_s2network(self):
        with torch_default_dtype(torch.float64):
            mul = 3
            Rs_in = [(mul, l) for l in range(3 + 1)]
            Rs_out = [(mul, l) for l in range(3 + 1)]

            net = S2Network(Rs_in, mul, lmax=4, Rs_out=Rs_out)

            abc = o3.rand_angles()
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(Rs_out, *abc)

            fea = torch.randn(10, rs.dim(Rs_in))

            x1 = torch.einsum("ij,zj->zi", D_out, net(fea))
            x2 = net(torch.einsum("ij,zj->zi", D_in, fea))
            self.assertLess((x1 - x2).norm(), 1e-3 * x1.norm())


if __name__ == '__main__':
    unittest.main()
