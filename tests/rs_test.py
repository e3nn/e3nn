# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, rs


class Tests(unittest.TestCase):

    def test_conventionRs(self):
        Rs = [(1, 0)]
        Rs_out = rs.convention(Rs)
        self.assertSequenceEqual(Rs_out, [(1, 0, 0)])
        Rs = [(1, 0), (2, 0)]
        Rs_out = rs.convention(Rs)
        self.assertSequenceEqual(Rs_out, [(1, 0, 0), (2, 0, 0)])

    def test_simplifyRs(self):
        Rs = [(1, 0), (2, 0)]
        Rs_out = rs.simplify(Rs)
        self.assertSequenceEqual(Rs_out, [(3, 0, 0)])

    def test_irrep_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(rs.irrep_dim(Rs) == 1 + 3 + 5)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(rs.irrep_dim(Rs) == 1 + 1 + 1)

    def test_mul_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(rs.mul_dim(Rs) == 6)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(rs.mul_dim(Rs) == 6)

    def test_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(rs.dim(Rs) == 1 * 1 + 3 * 3 + 2 * 5)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(rs.dim(Rs) == 1 * 1 + 3 * 1 + 2 * 1)

    def test_map_irrep_to_Rs(self):
        with o3.torch_default_dtype(torch.float64):
            Rs = [(3, 0)]
            mapping_matrix = rs.map_irrep_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.ones(3, 1)))

            Rs = [(1, 0), (1, 1), (1, 2)]
            mapping_matrix = rs.map_irrep_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.eye(1 + 3 + 5)))

    def test_map_mul_to_Rs(self):
        with o3.torch_default_dtype(torch.float64):
            Rs = [(3, 0)]
            mapping_matrix = rs.map_mul_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.eye(3)))

            Rs = [(1, 0), (1, 1), (1, 2)]
            mapping_matrix = rs.map_mul_to_Rs(Rs)
            check_matrix = torch.zeros(1 + 3 + 5, 3)
            check_matrix[0, 0] = 1.
            check_matrix[1:4, 1] = 1.
            check_matrix[4:, 2] = 1.
            self.assertTrue(torch.allclose(mapping_matrix, check_matrix))

    def test_elementwise_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = rs.elementwise_tensor_product(Rs_1, Rs_2)
        mul = rs.ElementwiseTensorProduct(Rs_1, Rs_2)

        x1 = torch.randn(1, rs.dim(Rs_1))
        x2 = torch.randn(1, rs.dim(Rs_2))

        y1 = mul(x1, x2)
        y2 = torch.einsum('zi,zj->ijz', x1, x2)
        y2 = m @ y2.reshape(-1, y2.shape[2])
        y2 = y2.T

        self.assertEqual(rs.dim(Rs_out), y1.shape[1])
        self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())

    def test_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = rs.tensor_product(Rs_1, Rs_2, o3.selection_rule, sorted=True)
        mul = rs.TensorProduct(Rs_1, Rs_2, o3.selection_rule)

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

                n = rs.dim(Rs_out)
                I = torch.eye(n, dtype=Q.dtype())

                d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                d = ((Q.t() @ Q).to_dense() - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

    def test_tensor_square_equivariance(self):
        with o3.torch_default_dtype(torch.float64):
            Rs_in = [(3, 0), (2, 1), (5, 2)]

            sq = rs.TensorSquare(Rs_in, o3.selection_rule)

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
                _, Q = rs.tensor_product(Rs_in1, o3.selection_rule, Rs_out)

                n = rs.dim(Rs_out)
                I = torch.eye(n, dtype=Q.dtype())

                x = Q @ Q.t()
                x = x.to_dense()
                d = (x - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)


def test_reduce_tensor_Levi_Civita_symbol():
    Rs, Q = rs.reduce_tensor('ijk=-ikj=-jik', i=[(1, 1)])
    assert Rs == [(1, 0, 0)]
    r = o3.rand_angles()
    D = o3.irr_repr(1, *r)
    Q = Q.reshape(3, 3, 3)
    Q1 = torch.einsum('li,mj,nk,ijk', D, D, D, Q)
    assert (Q1 - Q).abs().max() < 1e-10


def test_reduce_tensor_antisymmetric_L2():
    Rs, Q = rs.reduce_tensor('ijk=-ikj=-jik', i=[(1, 2)])
    assert Rs[0] == (1, 1, 0)
    q = Q[:3].reshape(3, 5, 5, 5)

    r = o3.rand_angles()
    D1 = o3.irr_repr(1, *r)
    D2 = o3.irr_repr(2, *r)
    Q1 = torch.einsum('il,jm,kn,zijk->zlmn', D2, D2, D2, q)
    Q2 = torch.einsum('yz,zijk->yijk', D1, q)

    assert (Q1 - Q2).abs().max() < 1e-10
    assert (q + q.transpose(1, 2)).abs().max() < 1e-10
    assert (q + q.transpose(1, 3)).abs().max() < 1e-10
    assert (q + q.transpose(3, 2)).abs().max() < 1e-10


def test_reduce_tensor_elasticity_tensor():
    Rs, _Q = rs.reduce_tensor('ijkl=jikl=klij', i=[(1, 1)])
    assert rs.dim(Rs) == 21


if __name__ == '__main__':
    unittest.main()
