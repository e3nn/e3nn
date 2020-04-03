# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import math
import unittest
from functools import partial

import torch

from e3nn import o3, rs


class Tests(unittest.TestCase):

    def test_decomposition_spherical_harmonics(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 4
            beta = torch.linspace(1e-3, math.pi - 1e-3, 100).view(1, -1)
            alpha = torch.linspace(0, 2 * math.pi, 100).view(-1, 1)
            Y1 = o3.spherical_harmonics_alpha_part(lmax, alpha) * o3.spherical_harmonics_beta_part(lmax, beta.cos())
            Y2 = o3.spherical_harmonics([l for l in range(lmax + 1)], alpha, beta)
            Y2 = torch.einsum('ulmi,iab->lmab', o3.spherical_harmonics_expand_matrix(1, lmax), Y2)
            self.assertLess((Y1 - Y2).abs().max(), 1e-10)

    def test_sh_is_in_irrep(self):
        with o3.torch_default_dtype(torch.float64):
            for l in range(4 + 1):
                a, b = 3.14 * torch.rand(2)  # works only for beta in [0, pi]
                Y = o3.spherical_harmonics(l, a, b) * math.sqrt(4 * math.pi) / math.sqrt(2 * l + 1) * (-1) ** l
                D = o3.irr_repr(l, a, b, 0)
                self.assertLess((Y - D[:, l]).norm(), 1e-10)

    def test_sh_cuda_single(self):
        if torch.cuda.is_available():
            with o3.torch_default_dtype(torch.float64):
                for l in range(10 + 1):
                    x = torch.randn(10, 3)
                    x_cuda = x.cuda()
                    Y1 = o3.spherical_harmonics_xyz(l, x)
                    Y2 = o3.spherical_harmonics_xyz(l, x_cuda).cpu()
                    self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_single skipped!")

    def test_sh_cuda_ordered_full(self):
        if torch.cuda.is_available():
            with o3.torch_default_dtype(torch.float64):
                l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                x = torch.randn(10, 3)
                x_cuda = x.cuda()
                Y1 = o3.spherical_harmonics_xyz(l, x)
                Y2 = o3.spherical_harmonics_xyz(l, x_cuda).cpu()
                self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_ordered_full skipped!")

    def test_sh_cuda_ordered_partial(self):
        if torch.cuda.is_available():
            with o3.torch_default_dtype(torch.float64):
                l = [0, 2, 5, 7, 10]
                x = torch.randn(10, 3)
                x_cuda = x.cuda()
                Y1 = o3.spherical_harmonics_xyz(l, x)
                Y2 = o3.spherical_harmonics_xyz(l, x_cuda).cpu()
                self.assertLess((Y1 - Y2).abs().max(), 1e-7)
        else:
            print("Cuda is not available! test_sh_cuda_ordered_partial skipped!")

    def test_sh_parity(self):
        """
        (-1)^l Y(x) = Y(-x)
        """
        with o3.torch_default_dtype(torch.float64):
            for l in range(7 + 1):
                x = torch.randn(3)
                Y1 = (-1) ** l * o3.spherical_harmonics_xyz(l, x)
                Y2 = o3.spherical_harmonics_xyz(l, -x)
                self.assertLess((Y1 - Y2).abs().max(), 1e-10 * Y1.abs().max())

    def test_sh_dirac(self):
        with o3.torch_default_dtype(torch.float64):
            for l in range(5):
                a = o3.spherical_harmonics_dirac(l, 1.2, 2.1)
                a = o3.spherical_harmonics_coeff_to_sphere(a, 1.2, 2.1)
                self.assertAlmostEqual(a.item(), 1)

    def test_sh_norm(self):
        with o3.torch_default_dtype(torch.float64):
            l_filter = list(range(15))
            Ys = [o3.spherical_harmonics_xyz(l, torch.randn(10, 3)) for l in l_filter]
            s = torch.stack([Y.pow(2).mean(0) for Y in Ys])
            d = s - 1 / (4 * math.pi)
            self.assertLess(d.pow(2).mean().sqrt(), 1e-10)

    def test_sh_closure(self):
        """
        integral of Ylm * Yjn = delta_lj delta_mn
        integral of 1 over the unit sphere = 4 pi
        """
        with o3.torch_default_dtype(torch.float64):
            x = torch.randn(200000, 3)
            Ys = [o3.spherical_harmonics_xyz(l, x) for l in range(0, 3 + 1)]
            for l1, Y1 in enumerate(Ys):
                for l2, Y2 in enumerate(Ys):
                    m = (Y1.view(2 * l1 + 1, 1, -1) * Y2.view(1, 2 * l2 + 1, -1)).mean(-1) * (4 * math.pi)
                    if l1 == l2:
                        i = torch.eye(2 * l1 + 1)
                        self.assertLess((m - i).pow(2).max(), 1e-4)
                    else:
                        self.assertLess(m.pow(2).max(), 1e-4)

    def test_clebsch_gordan_orthogonal(self):
        with o3.torch_default_dtype(torch.float64):
            for l_out in range(3 + 1):
                for l_in in range(l_out, 4 + 1):
                    for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                        Q = o3.clebsch_gordan(l_f, l_in, l_out).reshape(2 * l_f + 1, -1)
                        e = (2 * l_f + 1) * Q @ Q.t()
                        d = e - torch.eye(2 * l_f + 1)
                        self.assertLess(d.pow(2).mean().sqrt(), 1e-10)

    def test_clebsch_gordan_sh_norm(self):
        with o3.torch_default_dtype(torch.float64):
            for l_out in range(3 + 1):
                for l_in in range(l_out, 4 + 1):
                    for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                        Q = o3.clebsch_gordan(l_out, l_in, l_f)
                        Y = o3.spherical_harmonics_xyz(l_f, torch.randn(1, 3)).view(2 * l_f + 1)
                        QY = math.sqrt(4 * math.pi) * Q @ Y
                        self.assertLess(abs(QY.norm() - 1), 1e-10)

    def test_rot_to_abc(self):
        with o3.torch_default_dtype(torch.float64):
            R = o3.rand_rot()
            abc = o3.rot_to_abc(R)
            R2 = o3.rot(*abc)
            d = (R - R2).norm() / R.norm()
            self.assertTrue(d < 1e-10, d)

    def test_wignerd(self):
        for l__ in range(7):
            self._test_is_representation(partial(o3.irr_repr, l__))

    def _test_is_representation(self, R):
        """
        R(Z(a1) Y(b1) Z(c1) Z(a2) Y(b2) Z(c2)) = R(Z(a1) Y(b1) Z(c1)) R(Z(a2) Y(b2) Z(c2))
        """
        with o3.torch_default_dtype(torch.float64):
            a1, b1, c1, a2, b2, c2 = torch.rand(6)

            r1 = R(a1, b1, c1)
            r2 = R(a2, b2, c2)

            a, b, c = o3.compose(a1, b1, c1, a2, b2, c2)
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
            with o3.torch_default_dtype(torch.float64):
                a, b = torch.rand(2)
                alpha, beta, gamma = torch.rand(3)

                ra, rb, _ = o3.compose(alpha, beta, gamma, a, b, 0)
                Yrx = o3.spherical_harmonics(order, ra, rb)

                Y = o3.spherical_harmonics(order, a, b)
                DrY = o3.irr_repr(order, alpha, beta, gamma) @ Y

                self.assertLess((Yrx - DrY).abs().max(), 1e-10 * Y.abs().max())

    def test_xyz_vector_basis_to_spherical_basis(self, ):
        with o3.torch_default_dtype(torch.float64):
            A = o3.xyz_vector_basis_to_spherical_basis()

            a, b, c = torch.rand(3)

            r1 = A.t() @ o3.irr_repr(1, a, b, c) @ A
            r2 = o3.rot(a, b, c)

            self.assertLess((r1 - r2).abs().max(), 1e-10)

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


if __name__ == '__main__':
    unittest.main()
