# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, no-value-for-parameter
import unittest
from functools import partial

import torch

from e3nn import o3


class Tests(unittest.TestCase):

    def test_irrep_closure_1(self):
        rots = [o3.rand_angles() for _ in range(10000)]
        Us = [torch.stack([o3.irr_repr(l, *abc) for abc in rots]) for l in range(3 + 1)]
        for l1, U1 in enumerate(Us):
            for l2, U2 in enumerate(Us):
                m = torch.einsum('zij,zkl->zijkl', U1, U2).mean(0).reshape((2 * l1 + 1)**2, (2 * l2 + 1)**2)
                if l1 == l2:
                    i = torch.eye((2 * l1 + 1)**2)
                    self.assertLess((m.mul(2 * l1 + 1) - i).abs().max(), 0.1)
                else:
                    self.assertLess(m.abs().max(), 0.1)

    def test_irrep_closure_2(self):
        r1, r2 = (0, 0.2, 0), (0.1, 0.4, 1.5)  # two random rotations
        a = sum((o3.irr_repr(l, *r1) * o3.irr_repr(l, *r2)).sum() for l in range(12 + 1))
        b = sum((o3.irr_repr(l, *r1) * o3.irr_repr(l, *r1)).sum() for l in range(12 + 1))
        self.assertLess(a, b / 100)

    def test_wigner_3j_orthogonal(self):
        with o3.torch_default_dtype(torch.float64):
            for l_out in range(3 + 1):
                for l_in in range(l_out, 4 + 1):
                    for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                        Q = o3.wigner_3j(l_f, l_in, l_out).reshape(2 * l_f + 1, -1)
                        e = (2 * l_f + 1) * Q @ Q.t()
                        d = e - torch.eye(2 * l_f + 1)
                        self.assertLess(d.pow(2).mean().sqrt(), 1e-10)

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
            g1 = o3.rand_angles()
            g2 = o3.rand_angles()

            g12 = o3.compose(*g1, *g2)
            D12 = R(*g12)

            D1D2 = R(*g1) @ R(*g2)

            self.assertLess((D12 - D1D2).abs().max(), 1e-10 * D12.abs().max())

    def test_xyz_to_irreducible_basis(self, ):
        with o3.torch_default_dtype(torch.float64):
            A = o3.xyz_to_irreducible_basis()

            a, b, c = torch.rand(3)

            r1 = A.t() @ o3.irr_repr(1, a, b, c) @ A
            r2 = o3.rot(a, b, c)

            assert torch.allclose(r1, r2)

    def test_derivative_irr_repr(self):
        with o3.torch_default_dtype(torch.float64):
            l = 4
            angles = o3.rand_angles()
            da, db, dc = o3.derivative_irr_repr(l, *angles)
            h = 1e-7
            da1 = (o3.irr_repr(l, angles[0] + h, angles[1], angles[2]) - o3.irr_repr(l, *angles)) / h
            db1 = (o3.irr_repr(l, angles[0], angles[1] + h, angles[2]) - o3.irr_repr(l, *angles)) / h
            dc1 = (o3.irr_repr(l, angles[0], angles[1], angles[2] + h) - o3.irr_repr(l, *angles)) / h
            self.assertLess((da1 - da).abs().max(), 1e-5)
            self.assertLess((db1 - db).abs().max(), 1e-5)
            self.assertLess((dc1 - dc).abs().max(), 1e-5)

    def test_kron(self):
        with o3.torch_default_dtype(torch.float64):
            m1 = torch.randn(4, 4)
            m2 = torch.randn(3, 5)
            m3 = torch.randn(6, 6)

            x1 = o3.kron(m1, m2, m3)
            x2 = o3.kron(m1, o3.kron(m2, m3))
            assert torch.allclose(x1, x2)

    def test_xyz3x3_to_irreducible_basis(self):
        o3.xyz3x3_to_irreducible_basis()

    def test_intertwiners(self):
        with o3.torch_default_dtype(torch.float64):
            A = o3.intertwiners(partial(o3.irr_repr, 1), partial(o3.irr_repr, 1))
            assert A.shape[0] == 1

    def test_reduce(self):
        with o3.torch_default_dtype(torch.float64):
            A = torch.randn(3 + 3 + 5, 3 + 3 + 5)
            torch.nn.init.orthogonal_(A)

            def D(*angles):
                d = o3.direct_sum(
                    o3.irr_repr(1, *angles),
                    o3.irr_repr(1, *angles),
                    o3.irr_repr(2, *angles),
                )
                return A @ d @ torch.inverse(A)

            n, _, _ = o3.reduce(D, partial(o3.irr_repr, 1))
            assert n == 2


if __name__ == '__main__':
    unittest.main()
