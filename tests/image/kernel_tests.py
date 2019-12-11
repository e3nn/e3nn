# pylint: disable=C,E1101,E1102
import unittest
from functools import partial

import torch

from e3nn.image.kernel import (SE3Kernel, check_basis_equivariance,
                                 cube_basis_kernels, gaussian_window)
from e3nn.util.default_dtype import torch_default_dtype


class Tests(unittest.TestCase):
    def test_kij_is_none(self):
        kernel = SE3Kernel([(1, 0)], [(1, 0), (1, 5)], 3)
        kernel.forward()

    def test_basis_equivariance(self):
        with torch_default_dtype(torch.float64):
            basis = cube_basis_kernels(4 * 5, 2, 2, partial(gaussian_window, radii=[5], J_max_list=[999], sigma=2))
            overlaps = check_basis_equivariance(basis, 2, 2, *torch.rand(3))
            self.assertTrue(overlaps.gt(0.98).all(), overlaps)

unittest.main()
