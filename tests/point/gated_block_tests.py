# pylint: disable=invalid-name, missing-docstring, no-member, line-too-long
import unittest
from functools import partial

import torch

from e3nn.non_linearities import GatedBlock, rescaled_act
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.radial import ConstantRadialModel
from e3nn.SO3 import direct_sum, irr_repr, rot
from e3nn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test1(self):
        with torch_default_dtype(torch.float64):
            Rs_in = [(3, 0), (3, 1), (2, 0), (1, 2)]
            Rs_out = [(3, 0), (3, 1), (1, 2), (3, 0)]

            f = GatedBlock(Rs_out, rescaled_act.Softplus(beta=5), rescaled_act.sigmoid)

            K = partial(Kernel, RadialModel=ConstantRadialModel)
            c = Convolution(K, Rs_in, f.Rs_in)

            abc = torch.randn(3)
            D_in = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_in for _ in range(mul)])
            D_out = direct_sum(*[irr_repr(l, *abc) for mul, l in Rs_out for _ in range(mul)])

            x = torch.randn(1, 5, sum(mul * (2 * l + 1) for mul, l in Rs_in))
            geo = torch.randn(1, 5, 3)

            rx = torch.einsum("ij,zaj->zai", (D_in, x))
            rgeo = geo @ rot(*abc).t()

            y = f(c(x, geo), dim=2)
            ry = torch.einsum("ij,zaj->zai", (D_out, y))

            self.assertLess((f(c(rx, rgeo)) - ry).norm(), 1e-10 * ry.norm())


unittest.main()
