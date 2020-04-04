# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, rs, soft


class Tests(unittest.TestCase):

    def test_inverse(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            res = 50

            to = soft.ToSOFT(lmax, res)
            fr = soft.FromSOFT(res, lmax)

            sig = torch.randn(10, (lmax + 1) ** 2)
            self.assertLess((fr(to(sig)) - sig).abs().max(), 1e-5)

            s = to(sig)
            self.assertLess((to(fr(s)) - s).abs().max(), 1e-5)

    def test_normalization(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            res = 20

            for normalization in ['component', 'norm']:
                to = soft.ToSOFT(lmax, res, normalization=normalization)
                x = rs.randn(50, [(1, l) for l in range(lmax + 1)], normalization=normalization)
                y = to(x)

                self.assertAlmostEqual(y.var().item(), 1, delta=0.2)


if __name__ == '__main__':
    unittest.main()
