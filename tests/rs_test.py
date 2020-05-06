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

    def test_tensor_product_sparse_vs_dense(self):
        import numpy as np
        Rs_in1 = [(1, 1), (1, 3)]
        Rs_in2 = [(4, 1)]
        Rs_out, Q = rs._tensor_product_in_in(Rs_in1, Rs_in2, o3.selection_rule, "norm", sorted=False)
        Rs_out, index, value = rs._tensor_product_in_in_sparse(Rs_in1, Rs_in2, o3.selection_rule, "norm")
        print("num coeff in Q: ", np.array(Q.shape).prod(), "num coeff in value: ", value.shape[0])

        for i, v in zip (index, value):
            assert torch.allclose(Q[i[0], i[1], i[2]], v)

        Rs_in1 = [(2, 1)]
        Rs_out = [(3, 1), (1, 3)]
        Rs_in2, Q = rs._tensor_product_in_out(Rs_in1, o3.selection_rule, Rs_out, "norm", sorted=False)
        Rs_in2, index, value = rs._tensor_product_in_out_sparse(Rs_in1, o3.selection_rule, Rs_out, "norm")
        print("num coeff in Q: ", np.array(Q.shape).prod(), "num coeff in value: ", value.shape[0])

        for i, v in zip (index, value):
            assert torch.allclose(Q[i[0], i[1], i[2]], v)

if __name__ == '__main__':
    unittest.main()
