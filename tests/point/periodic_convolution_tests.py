# pylint: disable=C,E1101,E1102
import unittest

import torch
from functools import partial
from e3nn.point.operations import PeriodicConvolution
from e3nn.kernel import Kernel
from e3nn.radial import ConstantRadialModel


class Tests(unittest.TestCase):
    def test1(self):
        torch.set_default_dtype(torch.float64)
        import pymatgen
        lattice = pymatgen.Lattice.cubic(1.0)

        Rs_in = [(2, 0), (0, 1), (2, 2)]
        Rs_out = [(2, 0), (2, 1), (2, 2)]
        max_radius = 3.0
        K = partial(Kernel, RadialModel=ConstantRadialModel)
        m = PeriodicConvolution(K, Rs_in, Rs_out, max_radius=max_radius)
        n = sum(mul * (2 * l + 1) for mul, l in Rs_in)

        x = torch.randn(2, 3, n)
        g = torch.randn(2, 3, 3)
        m(x, g, lattice, max_radius)


unittest.main()
