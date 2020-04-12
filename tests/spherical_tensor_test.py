# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, spherical_tensor


class Tests(unittest.TestCase):
    def test_sh_dirac(self):
        with o3.torch_default_dtype(torch.float64):
            for l in range(5):
                a = spherical_tensor.spherical_harmonics_dirac(l, 1.2, 2.1)
                a = spherical_tensor.spherical_harmonics_coeff_to_sphere(a, torch.tensor(1.2), torch.tensor(2.1))
                self.assertAlmostEqual(a.item(), 1)
