# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, rs, s2grid


class Tests(unittest.TestCase):

    def test_fft(self):
        with o3.torch_default_dtype(torch.float64):
            for lmax in [0, 1, 10, 20]:
                res = 2 * lmax + 1

                _betas, _alphas, _shb, sha = s2grid.spherical_harmonics_s2_grid(lmax, res, res)

                # orthogonal
                self.assertLess((sha @ sha.T / res - torch.eye(res)).abs().max(), 1e-10)
                self.assertLess((sha.T @ sha / res - torch.eye(res)).abs().max(), 1e-10)

                # forward
                x = torch.randn(res)

                y1 = x @ sha

                y2 = torch.fft(torch.stack([x, torch.zeros_like(x)], dim=-1), 1)
                y2 = torch.cat([2**0.5 * y2[lmax + 1:, 1], y2[:1, 0], 2**0.5 * y2[1:lmax + 1, 0]])

                y3 = torch.rfft(x, 1)
                y3 = torch.cat([-2**0.5 * y3[1:, 1].flip(0), y3[:1, 0], 2**0.5 * y3[1:, 0]])

                self.assertLess((y3 - y2).abs().max(), 1e-10 * y3.abs().max())
                self.assertLess((y1 - y2).abs().max(), 1e-10 * y1.abs().max())

                # backward
                y = torch.randn(res)
                x1 = sha @ y

                x2 = torch.stack([
                    torch.cat([y[lmax:lmax + 1], y[lmax + 1:] / 2**0.5]),
                    torch.cat([torch.zeros(1), -y[:lmax].flip(0) / 2**0.5]),
                ], dim=-1)
                x2 = torch.irfft(x2, 1) * res

                self.assertLess((x1 - x2).abs().max(), 1e-10 * x1.abs().max())

    def test_fft2(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            res = 31
            _betas, _alphas, _shb, sha = s2grid.spherical_harmonics_s2_grid(lmax, res, res)

            x = torch.randn(2 * lmax + 1)
            y1 = sha @ x
            y2 = s2grid.irfft(x, res)

            self.assertLess((y1 - y2).abs().max(), 1e-10 * y1.abs().max())

            x = torch.randn(res)
            y1 = x @ sha
            y2 = s2grid.rfft(x, lmax)

            self.assertLess((y1 - y2).abs().max(), 1e-10 * y1.abs().max())

    def test_inverse(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            for res in [(50, 75), (2 * lmax + 2, 2 * lmax + 1)]:
                for normalization in ['component', 'norm', 'none']:
                    to = s2grid.ToS2Grid(lmax, res, normalization=normalization)
                    fr = s2grid.FromS2Grid(res, lmax, normalization=normalization)

                    sig = rs.randn(10, [(1, l) for l in range(lmax + 1)])
                    self.assertLess((fr(to(sig)) - sig).abs().max(), 1e-5)

                    s = to(sig)
                    self.assertLess((to(fr(s)) - s).abs().max(), 1e-5)

    def test_inverse_different_ls(self):
        with o3.torch_default_dtype(torch.float64):
            lin = 5
            lout = 7
            res = (50, 60)

            for normalization in ['component', 'norm', 'none']:
                to = s2grid.ToS2Grid(lin, res, normalization=normalization)
                fr = s2grid.FromS2Grid(res, lout, lmax_in=lin, normalization=normalization)

                si = rs.randn(10, [(1, l) for l in range(lin + 1)])
                so = fr(to(si))
                so = so[:, :si.shape[1]]
                self.assertLess((so - si).abs().max(), 1e-5)

    def test_normalization(self):
        with o3.torch_default_dtype(torch.float64):
            lmax = 5
            res = (20, 30)

            for normalization in ['component', 'norm']:
                to = s2grid.ToS2Grid(lmax, res, normalization=normalization)
                x = rs.randn(50, [(1, l) for l in range(lmax + 1)], normalization=normalization)
                y = to(x)

                self.assertAlmostEqual(y.var().item(), 1, delta=0.2)


if __name__ == '__main__':
    unittest.main()
