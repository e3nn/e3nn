# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import SO3
from e3nn.tensor_product import ElementwiseTensorProduct, TensorProduct


class Tests(unittest.TestCase):

    def test_elementwise_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = SO3.elementwise_tensor_productRs(Rs_1, Rs_2)
        mul = ElementwiseTensorProduct(Rs_1, Rs_2)

        x1 = torch.randn(1, SO3.dimRs(Rs_1))
        x2 = torch.randn(1, SO3.dimRs(Rs_2))

        y1 = mul(x1, x2)
        y2 = torch.einsum('kij,zi,zj->zk', m, x1, x2)

        self.assertEqual(SO3.dimRs(Rs_out), y1.shape[1])
        self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())


    def test_tensor_product(self):
        torch.set_default_dtype(torch.float64)

        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = SO3.tensor_productRs(Rs_1, Rs_2)
        mul = TensorProduct(Rs_1, Rs_2)

        x1 = torch.randn(1, SO3.dimRs(Rs_1))
        x2 = torch.randn(1, SO3.dimRs(Rs_2))

        y1 = mul(x1, x2)
        y2 = torch.einsum('kij,zi,zj->zk', m, x1, x2)

        self.assertEqual(SO3.dimRs(Rs_out), y1.shape[1])
        self.assertLess((y1 - y2).abs().max(), 1e-7 * y1.abs().max())
