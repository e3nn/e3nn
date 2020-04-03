# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, soft


class Tests(unittest.TestCase):

    def test_inverse(self):
        with o3.torch_default_dtype(torch.float64):
            mul = 2
            lmax = 5
            res = 50

            to = soft.ToSOFT(mul, lmax, res)
            fr = soft.FromSOFT(mul, res, lmax)

            sig = torch.randn(10, mul * (lmax + 1) ** 2)
            self.assertLess((fr(to(sig)) - sig).abs().max(), 1e-5)

            s = to(sig)
            self.assertLess((to(fr(s)) - s).abs().max(), 1e-5)

    def test_normalization(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            res = 20

            to = soft.ToSOFT(1, lmax, res)

            sig = torch.randn(50, (lmax + 1) ** 2)
            self.assertAlmostEqual(to(sig).var().item(), 1, delta=0.2)


if __name__ == '__main__':
    unittest.main()
