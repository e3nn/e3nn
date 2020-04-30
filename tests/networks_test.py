# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
import unittest

import torch

from e3nn import o3
from e3nn import rs
from e3nn.networks import GatedConvParityNetwork, GatedConvNetwork


class Tests(unittest.TestCase):
    def test_parity_network(self):
        torch.set_default_dtype(torch.float64)

        lmax = 3
        Rs = [(1, l, 1) for l in range(lmax + 1)]
        model = GatedConvParityNetwork(Rs, 4, Rs, lmax, feature_product=True)

        features = rs.randn(1, 4, Rs)
        geometry = torch.randn(1, 4, 3)

        output = model(features, geometry)

        angles = o3.rand_angles()
        D = rs.rep(Rs, *angles, 1)
        R = -o3.rot(*angles)
        ein = torch.einsum
        output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

        self.assertLess((output - output2).abs().max(), 1e-10 * output.abs().max())

    def test_network(self):
        torch.set_default_dtype(torch.float64)

        lmax = 3
        Rs = [(1, l) for l in range(lmax + 1)]
        model = GatedConvNetwork(Rs, 4 * Rs, Rs, lmax, feature_product=True)

        features = rs.randn(1, 4, Rs)
        geometry = torch.randn(1, 4, 3)

        output = model(features, geometry)

        angles = o3.rand_angles()
        D = rs.rep(Rs, *angles)
        R = o3.rot(*angles)
        ein = torch.einsum
        output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

        self.assertLess((output - output2).abs().max(), 1e-10 * output.abs().max())


if __name__ == '__main__':
    unittest.main()
