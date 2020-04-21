# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3
import e3nn.spherical_tensor as sphten


class Tests(unittest.TestCase):
    def test_sh_dirac(self):
        with o3.torch_default_dtype(torch.float64):
            for l in range(5):
                a = sphten.spherical_harmonics_dirac(l, torch.tensor(1.2), torch.tensor(2.1))
                a = sphten.spherical_harmonics_coeff_to_sphere(a, torch.tensor(1.2), torch.tensor(2.1))
                self.assertAlmostEqual(a.item(), 1)
    
    def test_projection(self):
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sphten.projection(coords, lmax)

    def test_adjusted_projection(self):
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sphten.adjusted_projection(coords, lmax)

    def test_plot_on_grid(self):
        pass

class SphericalTensorTests(unittest.TestCase):
    def test_SphericalTensor(self):
        torch.set_default_dtype(torch.float64)
        lmax = 6
        mul = 1
        sphten.SphericalTensor(torch.randn((lmax + 1) ** 2), mul, lmax)
        mul = 3
        sphten.SphericalTensor(torch.randn((lmax + 1) ** 2), mul, lmax)

    def test_SphericalTensor_from_geometry(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sphten.SphericalTensor.from_geometry(coords, lmax)

    def test_SphericalTensor_from_geometry_with_radial(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        radial_model = lambda x: torch.ones_like(x).unsqueeze(-1)
        sphten.SphericalTensor.from_geometry_with_radial(coords, radial_model, lmax)

    def test_SphericalTensor_sph_norm(self):
        torch.set_default_dtype(torch.float64)
        lmax = 6
        mul = 1
        sph = sphten.SphericalTensor(torch.randn((lmax + 1) ** 2), mul, lmax)
        sph.sph_norm()

        mul = 3
        sph = sphten.SphericalTensor(torch.randn((lmax + 1) ** 2), mul, lmax)
        sph.sph_norm()

    def test_SphericalTensor_plot(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sph = sphten.SphericalTensor.from_geometry(coords, lmax)

        n = 16
        r, f = sph.plot(n=n)
        assert list(r.shape) == [n, n + 1, 3]
        assert list(f.shape) == [n, n + 1]

    def test_SphericalTensor_plot_with_radial(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        radial_model = lambda x: torch.ones_like(x).unsqueeze(-1)
        sph = sphten.SphericalTensor.from_geometry_with_radial(coords, radial_model, lmax)
        
        n = 16
        r, f = sph.plot_with_radial(box_length=3.0, n=n)
        assert list(r.shape) == [n ** 3, 3]
        print(n, f.shape)
        assert list(f.shape) == [n ** 3]

if __name__ == '__main__':
    unittest.main()
