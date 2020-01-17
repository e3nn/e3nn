# pylint: disable=C,E1101,E1102
import unittest

import torch
from e3nn.point.kernel import Kernel, KernelFn
from e3nn.point.radial import ConstantRadialModel


class Tests(unittest.TestCase):
    def test1(self):
        torch.set_default_dtype(torch.float64)
        Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
        Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]
        kernel = Kernel(Rs_in, Rs_out, ConstantRadialModel)

        n_path = 0
        for mul_out, l_out, p_out in kernel.Rs_out:
            for mul_in, l_in, p_in in kernel.Rs_in:
                l_filters = kernel.get_l_filters(l_in, p_in, l_out, p_out)
                n_path += mul_out * mul_in * len(l_filters)

        for gy, gc in [(True, True), (True, False), (False, True)]:
            r = torch.randn(2, 3)
            radii = r.norm(2, dim=1)  # [batch]
            Ys = kernel.sh(kernel.set_of_l_filters, r)  # [l_filter * m_filter, batch]
            Ys = Ys.clone().detach().requires_grad_(gy)
            coefficients = torch.randn(2, n_path, requires_grad=gc)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]
            norm_coef = kernel.norm_coef
            norm_coef = norm_coef[:, :, (radii == 0).type(torch.long)]  # [l_out, l_in, batch]

            inputs = (
                Ys, coefficients, norm_coef, kernel.Rs_in, kernel.Rs_out, kernel.get_l_filters, kernel.set_of_l_filters
            )
            self.assertTrue(torch.autograd.gradcheck(KernelFn.apply, inputs))


unittest.main()
