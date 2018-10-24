import unittest, torch
import torch.nn as nn

from se3cnn import SE3Convolution

class Tests(unittest.TestCase):
    def _test_equivariance(self, f):
        torch.set_default_dtype(torch.float64)

        def rotate(t):
            # rotate 90 degrees in plane of axes 2 and 3
            return torch.flip(t, (2, )).transpose(2, 3)

        def unrotate(t):
            # undo the rotation by 3 more rotations
            return rotate(rotate(rotate(t)))

        inp = torch.randn(2, 1, 16, 16, 16)
        inp_r = rotate(inp)

        diff_inp = (inp - unrotate(inp_r)).abs().max().item()
        self.assertLess(diff_inp, 1e-10) # sanity check

        out = f(inp)
        out_r = f(inp_r)

        diff_out = (out - unrotate(out_r)).abs().max().item()
        self.assertLess(diff_out, 1e-10)

    def test_equivariance(self):
        f = torch.nn.Sequential(
            SE3Convolution([(1, 0)], [(2, 0), (2, 1), (1, 2)], size=5),
            SE3Convolution([(2, 0), (2, 1), (1, 2)], [(1, 0)], size=5),
        ).to(torch.float64)

        self._test_equivariance(f)


    def test_normalization(self):
        batch = 3
        size = 5
        input_size = 15
        Rs_in = [(2, 0), (1, 1), (3, 4)]
        Rs_out = [(2, 0), (2, 1), (1, 2)]

        conv = SE3Convolution(Rs_in, Rs_out, size)

        n_out = sum([m * (2 * l + 1) for m, l in Rs_out])
        n_in = sum([m * (2 * l + 1) for m, l in Rs_in])

        x = torch.randn(batch, n_in, input_size, input_size, input_size)
        y = conv(x)

        self.assertEqual(y.size(1), n_out)

        y_mean, y_std = y.mean().item(), y.std().item()

        self.assertLess(abs(y_mean), 0.1)
        self.assertLess(abs(y_std - 1), 0.3)

    def test_combination_gradient(self):
        Rs_in = [(1, 0), (1, 1)]
        Rs_out = [(1, 0)]
        size = 5

        conv = SE3Convolution(Rs_in, Rs_out, size).type(torch.float64)

        x = torch.rand(1, sum(m * (2 * l + 1) for m, l in Rs_in), 6, 6, 6,
                       requires_grad=True, dtype=torch.float64)

        self.assertTrue(torch.autograd.gradcheck(conv, (x, ), eps=1))

    @unittest.skipUnless(torch.__version__.startswith('1'), "jit requires >1.0")
    def test_tracing_jit(self):
        f = torch.nn.Sequential(
            SE3Convolution([(1, 0)], [(2, 0), (2, 1), (1, 2)], size=5),
            SE3Convolution([(2, 0), (2, 1), (1, 2)], [(1, 0)], size=5),
        )

        inp = torch.randn(2, 1, 16, 16, 16)

        traced = torch.jit.trace(f, inp)

        self.assertTrue(torch.allclose(f(inp), traced(inp)))

    @unittest.skipUnless(
        torch.cuda.device_count() > 1,
        "need at least 2 GPUs to meaningfully test DataParallel"
    )
    def test_data_parallel(self):
        f = torch.nn.Sequential(
            SE3Convolution([(1, 0)], [(2, 0), (2, 1), (1, 2)], size=5),
            SE3Convolution([(2, 0), (2, 1), (1, 2)], [(1, 0)], size=5),
        ).cuda()

        inp = torch.randn(2, 1, 16, 16, 16).cuda()

        parallel = nn.DataParallel(f).cuda()

        self.assertTrue(torch.allclose(f(inp), parallel(inp)))
        

unittest.main()
