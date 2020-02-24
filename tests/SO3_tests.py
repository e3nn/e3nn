# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import math
import unittest
from functools import partial

import torch

from e3nn import SO3


class Tests(unittest.TestCase):

    def test_sh_cuda_single(self):
        if torch.cuda.is_available():
            with SO3.torch_default_dtype(torch.float64):
                for l in range(10 + 1):
                    x = torch.randn(10, 3)
                    x_cuda = x.cuda()
                    Y1 = SO3.spherical_harmonics_xyz(l, x)
                    Y2 = SO3.spherical_harmonics_xyz(l, x_cuda).cpu()
                    self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_single skipped!")

    def test_sh_cuda_ordered_full(self):
        if torch.cuda.is_available():
            with SO3.torch_default_dtype(torch.float64):
                l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                x = torch.randn(10, 3)
                x_cuda = x.cuda()
                Y1 = SO3.spherical_harmonics_xyz(l, x)
                Y2 = SO3.spherical_harmonics_xyz(l, x_cuda).cpu()
                self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_ordered_full skipped!")

    def test_sh_cuda_ordered_partial(self):
        if torch.cuda.is_available():
            with SO3.torch_default_dtype(torch.float64):
                l = [0, 2, 5, 7, 10]
                x = torch.randn(10, 3)
                x_cuda = x.cuda()
                Y1 = SO3.spherical_harmonics_xyz(l, x)
                Y2 = SO3.spherical_harmonics_xyz(l, x_cuda).cpu()
                self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_ordered_partial skipped!")

    def test_sh_parity(self):
        """
        (-1)^l Y(x) = Y(-x)
        """
        with SO3.torch_default_dtype(torch.float64):
            for l in range(7 + 1):
                x = torch.randn(3)
                Y1 = (-1) ** l * SO3.spherical_harmonics_xyz(l, x)
                Y2 = SO3.spherical_harmonics_xyz(l, -x)
                self.assertLess((Y1 - Y2).abs().max(), 1e-10 * Y1.abs().max())

    def test_sh_dirac(self):
        with SO3.torch_default_dtype(torch.float64):
            for l in range(5):
                a = SO3.spherical_harmonics_dirac(l, 1.2, 2.1)
                a = SO3.spherical_harmonics_coeff_to_sphere(a, 1.2, 2.1)
                self.assertAlmostEqual(a.item(), 1)

    def test_sh_norm(self):
        with SO3.torch_default_dtype(torch.float64):
            l_filter = list(range(15))
            Ys = [SO3.spherical_harmonics_xyz(l, torch.randn(10, 3)) for l in l_filter]
            s = torch.stack([Y.pow(2).mean(0) for Y in Ys])
            d = s - 1 / (4 * math.pi)
            self.assertLess(d.pow(2).mean().sqrt(), 1e-10)

    def test_sh_closure(self):
        """
        integral of Ylm * Yjn = delta_lj delta_mn
        integral of 1 over the unit sphere = 4 pi
        """
        with SO3.torch_default_dtype(torch.float64):
            for l1 in range(0, 3 + 1):
                for l2 in range(l1, 3 + 1):
                    x = torch.randn(200000, 3)
                    Y1 = SO3.spherical_harmonics_xyz(l1, x)
                    Y2 = SO3.spherical_harmonics_xyz(l2, x)
                    x = (Y1.view(2 * l1 + 1, 1, -1) * Y2.view(1, 2 * l2 + 1, -1)).mean(-1) * (4 * math.pi)
                    if l1 == l2:
                        i = torch.eye(2 * l1 + 1)
                        self.assertLess((x - i).pow(2).max(), 1e-4)
                    else:
                        self.assertLess(x.pow(2).max(), 1e-4)

    def test_clebsch_gordan_orthogonal(self):
        with SO3.torch_default_dtype(torch.float64):
            for l_out in range(6):
                for l_in in range(6):
                    for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                        Q = SO3.clebsch_gordan(l_f, l_in, l_out).view(2 * l_f + 1, -1)
                        e = (2 * l_f + 1) * Q @ Q.t()
                        d = e - torch.eye(2 * l_f + 1)
                        self.assertLess(d.pow(2).mean().sqrt(), 1e-10)

    def test_clebsch_gordan_sh_norm(self):
        with SO3.torch_default_dtype(torch.float64):
            for l_out in range(6):
                for l_in in range(6):
                    for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                        Q = SO3.clebsch_gordan(l_out, l_in, l_f)
                        Y = SO3.spherical_harmonics_xyz(l_f, torch.randn(1, 3)).view(2 * l_f + 1)
                        QY = math.sqrt(4 * math.pi) * Q @ Y
                        self.assertLess(abs(QY.norm() - 1), 1e-10)

    def test_rot_to_abc(self):
        with SO3.torch_default_dtype(torch.float64):
            R = SO3.rand_rot()
            abc = SO3.rot_to_abc(R)
            R2 = SO3.rot(*abc)
            d = (R - R2).norm() / R.norm()
            self.assertTrue(d < 1e-10, d)

    def test_wignerd(self):
        for l__ in range(7):
            self._test_is_representation(partial(SO3.irr_repr, l__))

    def _test_is_representation(self, R):
        """
        R(Z(a1) Y(b1) Z(c1) Z(a2) Y(b2) Z(c2)) = R(Z(a1) Y(b1) Z(c1)) R(Z(a2) Y(b2) Z(c2))
        """
        with SO3.torch_default_dtype(torch.float64):
            a1, b1, c1, a2, b2, c2 = torch.rand(6)

            r1 = R(a1, b1, c1)
            r2 = R(a2, b2, c2)

            a, b, c = SO3.compose(a1, b1, c1, a2, b2, c2)
            r = R(a, b, c)

            r_ = r1 @ r2

            self.assertLess((r - r_).abs().max(), 1e-10 * r.abs().max())

    def test_spherical_harmonics(self):
        """
        This test tests that
        - irr_repr
        - compose
        - spherical_harmonics
        are compatible

        Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
        with x = Z(a) Y(b) eta
        """
        for order in range(7):
            with SO3.torch_default_dtype(torch.float64):
                a, b = torch.rand(2)
                alpha, beta, gamma = torch.rand(3)

                ra, rb, _ = SO3.compose(alpha, beta, gamma, a, b, 0)
                Yrx = SO3.spherical_harmonics(order, ra, rb)

                Y = SO3.spherical_harmonics(order, a, b)
                DrY = SO3.irr_repr(order, alpha, beta, gamma) @ Y

                self.assertLess((Yrx - DrY).abs().max(), 1e-10 * Y.abs().max())

    def test_xyz_vector_basis_to_spherical_basis(self, ):
        with SO3.torch_default_dtype(torch.float64):
            A = SO3.xyz_vector_basis_to_spherical_basis()

            a, b, c = torch.rand(3)

            r1 = A.t() @ SO3.irr_repr(1, a, b, c) @ A
            r2 = SO3.rot(a, b, c)

            self.assertLess((r1 - r2).abs().max(), 1e-10)

    def test_reduce_tensor_product(self):
        for Rs_i, Rs_j in [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(2, 0), (1, 1), (1, 3)])]:
            with SO3.torch_default_dtype(torch.float64):
                Rs, Q = SO3.tensor_productRs(Rs_i, Rs_j)

                abc = torch.rand(3, dtype=torch.float64)

                D_i = SO3.direct_sum(*[SO3.irr_repr(l, *abc) for mul, l in Rs_i for _ in range(mul)])
                D_j = SO3.direct_sum(*[SO3.irr_repr(l, *abc) for mul, l in Rs_j for _ in range(mul)])
                D = SO3.direct_sum(*[SO3.irr_repr(l, *abc) for mul, l, _ in Rs for _ in range(mul)])

                Q1 = torch.einsum("ijk,il->ljk", (Q, D))
                Q2 = torch.einsum("li,mj,kij->klm", (D_i, D_j, Q))

                d = (Q1 - Q2).pow(2).mean().sqrt() / Q1.pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                n = Q.size(0)
                M = Q.view(n, n)
                I = torch.eye(n, dtype=M.dtype)

                d = ((M @ M.t()) - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

                d = ((M.t() @ M) - I).pow(2).mean().sqrt()
                self.assertLess(d, 1e-10)

    def test_conventionRs(self):
        Rs = [(1, 0)]
        Rs_out = SO3.conventionRs(Rs)
        self.assertSequenceEqual(Rs_out, [(1, 0, 0)])
        Rs = [(1, 0), (2, 0)]
        Rs_out = SO3.conventionRs(Rs)
        self.assertSequenceEqual(Rs_out, [(1, 0, 0), (2, 0, 0)])

    def test_simplifyRs(self):
        Rs = [(1, 0), (2, 0)]
        Rs_out = SO3.simplifyRs(Rs)
        self.assertSequenceEqual(Rs_out, [(3, 0, 0)])

    def test_irrep_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(SO3.irrep_dimRs(Rs) == 1 + 3 + 5)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(SO3.irrep_dimRs(Rs) == 1 + 1 + 1)

    def test_mul_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(SO3.mul_dimRs(Rs) == 6)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(SO3.mul_dimRs(Rs) == 6)

    def test_dimRs(self):
        Rs = [(1, 0), (3, 1), (2, 2)]
        self.assertTrue(SO3.dimRs(Rs) == 1 * 1 + 3 * 3 + 2 * 5)
        Rs = [(1, 0), (3, 0), (2, 0)]
        self.assertTrue(SO3.dimRs(Rs) == 1 * 1 + 3 * 1 + 2 * 1)

    def test_map_irrep_to_Rs(self):
        with SO3.torch_default_dtype(torch.float64):
            Rs = [(3, 0)]
            mapping_matrix = SO3.map_irrep_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.ones(3, 1)))

            Rs = [(1, 0), (1, 1), (1, 2)]
            mapping_matrix = SO3.map_irrep_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.eye(1 + 3 + 5)))

    def test_map_mul_to_Rs(self):
        with SO3.torch_default_dtype(torch.float64):
            Rs = [(3, 0)]
            mapping_matrix = SO3.map_mul_to_Rs(Rs)
            self.assertTrue(torch.allclose(mapping_matrix, torch.eye(3)))

            Rs = [(1, 0), (1, 1), (1, 2)]
            mapping_matrix = SO3.map_mul_to_Rs(Rs)
            check_matrix = torch.zeros(1 + 3 + 5, 3)
            check_matrix[0, 0] = 1.
            check_matrix[1:4, 1] = 1.
            check_matrix[4:, 2] = 1.
            self.assertTrue(torch.allclose(mapping_matrix, check_matrix))


unittest.main()
