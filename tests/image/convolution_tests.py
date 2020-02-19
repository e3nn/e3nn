# pylint: disable=C,E1101,E1102
import unittest

import torch

from e3nn import SO3
from e3nn.image.convolution import Convolution


class Tests(unittest.TestCase):
    def _test_equivariance(self, f):
        def rotate(t):
            # rotate 90 degrees in plane of axes 2 and 3
            return torch.flip(t, (2, )).transpose(2, 3)

        def unrotate(t):
            # undo the rotation by 3 more rotations
            return rotate(rotate(rotate(t)))

        inp = torch.randn(2, 1, 16, 16, 16)
        inp_r = rotate(inp)

        diff_inp = (inp - unrotate(inp_r)).abs().max().item()
        self.assertLess(diff_inp, 1e-10)  # sanity check

        out = f(inp)
        out_r = f(inp_r)

        diff_out = (out - unrotate(out_r)).abs().max().item()
        self.assertLess(diff_out, 1e-10)

    def test_equivariance(self):
        torch.set_default_dtype(torch.float64)

        f = torch.nn.Sequential(
            Convolution([(1, 0)], [(2, 0), (2, 1), (1, 2)], size=5),
            Convolution([(2, 0), (2, 1), (1, 2)], [(1, 0)], size=5),
        ).to(torch.float64)

        self._test_equivariance(f)

    def _test_normalization(self, f):
        batch = 3
        size = 5
        input_size = 15
        Rs_in = [(10, 0), (10, 1), (10, 2)]
        Rs_out = [(2, 0), (2, 1), (2, 2)]

        conv = f(Rs_in, Rs_out, size)

        n_in = SO3.dimRs(Rs_in)
        n_out = SO3.dimRs(Rs_out)

        x = torch.randn(batch, n_in, input_size, input_size, input_size)
        y = conv(x)

        self.assertEqual(y.size(1), n_out)

        y_mean, y_std = y.mean().item(), y.std().item()

        self.assertAlmostEqual(y_mean, 0, delta=0.3)
        self.assertAlmostEqual(y_std, 1, delta=0.5)

    def test_normalization_conv(self):
        self._test_normalization(Convolution)


unittest.main()
