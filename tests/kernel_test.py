# pylint: disable=C,E1101,E1102
import unittest
from functools import partial

import torch

from e3nn import o3, rsh, rs
from e3nn.kernel import Kernel, KernelFn, GroupKernel
from e3nn.radial import ConstantRadialModel


class Tests(unittest.TestCase):
    def test1(self):
        """test gradients of the Kernel"""
        torch.set_default_dtype(torch.float64)
        Rs_in = [(1, 0), (1, 1), (1, 0), (1, 2)]
        Rs_out = [(1, 0), (1, 1), (1, 2), (1, 0)]
        kernel = Kernel(Rs_in, Rs_out, ConstantRadialModel, partial(o3.selection_rule_in_out_sh, lmax=1))

        n_path = 0
        for mul_out, l_out, p_out in kernel.Rs_out:
            for mul_in, l_in, p_in in kernel.Rs_in:
                l_filters = kernel.selection_rule(l_in, p_in, l_out, p_out)
                n_path += mul_out * mul_in * len(l_filters)

        r = torch.randn(2, 3)
        Y = rsh.spherical_harmonics_xyz(kernel.set_of_l_filters, r)  # [l_filter * m_filter, batch]
        Y = Y.clone().detach().requires_grad_(True)
        R = torch.randn(2, n_path, requires_grad=True)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        inputs = (
            Y, R, kernel.norm_coef, kernel.Rs_in, kernel.Rs_out, kernel.selection_rule, kernel.set_of_l_filters
        )
        self.assertTrue(torch.autograd.gradcheck(KernelFn.apply, inputs))


class TestCompare(unittest.TestCase):
    def setUp(self):
        super(TestCompare, self).setUp()
        torch.set_default_dtype(torch.float64)
        torch.backends.cudnn.deterministic = True
        self.Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
        self.Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        batch = 100
        atoms = 40
        self.geometry = torch.rand(batch, atoms, 3)

        self.msg = "Kernel parameters were not identical. This means the test cannot compare outputs."

    def test_compare_forward(self):
        for normalization in ["norm", "component"]:
            torch.manual_seed(0)
            K = Kernel(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            new_features = K(self.geometry)

            torch.manual_seed(0)
            K2 = Kernel(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = K2(self.geometry, custom_backward=True)

            assert all(torch.all(a == b) for a, b in zip(K.parameters(), K.parameters())), self.msg
            self.assertTrue(torch.allclose(new_features, check_new_features))


def test_group_kernel():
    kernel = partial(Kernel, RadialModel=ConstantRadialModel)
    Rs_in = [(5, 0, 1), (4, 1, -1)]
    Rs_out = [(3, 0, 1), (5, 1, -1)]
    groups = 4
    gkernel = GroupKernel(Rs_in, Rs_out, kernel, groups)

    N = 7
    input = torch.randn(N, 3)
    output = gkernel(input)
    assert output.dim() == 4  # [N, g, cout, cin]
    assert tuple(output.shape) == (N, groups, rs.dim(Rs_out), rs.dim(Rs_in))


if __name__ == '__main__':
    unittest.main()
