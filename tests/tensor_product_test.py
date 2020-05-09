# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import rs, o3
from e3nn.tensor_product import ElementwiseTensorProduct, TensorProduct, TensorSquare


class Tests(unittest.TestCase):

    def test_elementwise_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = rs.elementwise_tensor_product(Rs_1, Rs_2)
        mul = ElementwiseTensorProduct(Rs_1, Rs_2)

        x1 = torch.randn(1, rs.dim(Rs_1))
        x2 = torch.randn(1, rs.dim(Rs_2))

        y1 = mul(x1, x2)
        y2 = torch.einsum('kij,zi,zj->zk', m, x1, x2)

        self.assertEqual(rs.dim(Rs_out), y1.shape[1])
        self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())

    def test_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = rs.tensor_product(Rs_1, Rs_2, o3.selection_rule, sorted=True)
        mul = TensorProduct(Rs_1, Rs_2)

        x1 = rs.randn(1, Rs_1)
        x2 = rs.randn(1, Rs_2)

        y1 = mul(x1, x2)
        y2 = torch.einsum('zi,zj->ijz', x1, x2)
        y2 = (m @ y2.reshape(rs.dim(Rs_1) * rs.dim(Rs_2), -1)).T

        self.assertEqual(rs.dim(Rs_out), y1.shape[1])
        self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())

    def test_tensor_product_norm(self):
        for Rs_in1, Rs_in2 in [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(2, 0), (1, 1), (1, 3)])]:
            with o3.torch_default_dtype(torch.float64):
                Rs_out, Q = rs.tensor_product(Rs_in1, Rs_in2, o3.selection_rule)

                abc = torch.rand(3, dtype=torch.float64)

                D_in1 = rs.rep(Rs_in1, *abc)
                D_in2 = rs.rep(Rs_in2, *abc)
                D_out = rs.rep(Rs_out, *abc)

                Q1 = torch.einsum("ijk,il->ljk", (Q, D_out))
                Q2 = torch.einsum("li,mj,kij->klm", (D_in1, D_in2, Q))

                d = (Q1 - Q2).pow(2).mean().sqrt() / Q1.pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                n = Q.size(0)
                M = Q.reshape(n, n)
                I = torch.eye(n, dtype=M.dtype)

                d = ((M @ M.t()) - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                d = ((M.t() @ M) - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

    def test_tensor_square_equivariance(self):
        with o3.torch_default_dtype(torch.float64):
            Rs_in = [(3, 0), (2, 1), (5, 2)]

            sq = TensorSquare(Rs_in, o3.selection_rule)

            x = rs.randn(Rs_in)

            abc = o3.rand_angles()
            D_in = rs.rep(Rs_in, *abc)
            D_out = rs.rep(sq.Rs_out, *abc)

            y1 = sq(D_in @ x)
            y2 = D_out @ sq(x)

            self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())

    def test_tensor_square_norm(self):
        for Rs_in in [[(1, 0), (1, 1)]]:
            with o3.torch_default_dtype(torch.float64):
                Rs_out, Q = rs.tensor_square(Rs_in, o3.selection_rule, normalization='component', sorted=True)

                I1 = (Q @ Q.t()).to_dense()
                I2 = torch.eye(rs.dim(Rs_out))

                d = (I1 - I2).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

    def test_tensor_product_in_out_norm(self):
        for Rs_in1, Rs_out in [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(2, 0), (1, 1), (1, 3)])]:
            with o3.torch_default_dtype(torch.float64):
                Rs_in2, Q = rs.tensor_product(Rs_in1, o3.selection_rule, Rs_out)

                abc = torch.rand(3, dtype=torch.float64)

                D_in1 = rs.rep(Rs_in1, *abc)
                D_in2 = rs.rep(Rs_in2, *abc)
                D_out = rs.rep(Rs_out, *abc)

                Q1 = torch.einsum("ijk,il->ljk", (Q, D_out))
                Q2 = torch.einsum("li,mj,kij->klm", (D_in1, D_in2, Q))

                d = (Q1 - Q2).pow(2).mean().sqrt() / Q1.pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                n = Q.size(0)
                M = Q.reshape(n, -1)
                I = torch.eye(n, dtype=M.dtype)

                d = ((M @ M.t()) - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)


if __name__ == '__main__':
    unittest.main()
