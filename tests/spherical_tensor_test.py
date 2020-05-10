# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch

from e3nn import o3, rs
import e3nn.spherical_tensor as sphten


class Tests(unittest.TestCase):
    def test_sh_dirac(self):
        with o3.torch_default_dtype(torch.float64):
            for l in range(5):
                r = torch.randn(3)
                a = sphten.spherical_harmonics_dirac(r, l)
                v = sphten.SphericalTensor(a, l).signal_xyz(r)
                self.assertAlmostEqual(v.item(), 1)

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


class SphericalAndFourierTensorTests(unittest.TestCase):
    def test_SphericalTensor(self):
        torch.set_default_dtype(torch.float64)
        lmax = 6
        mul = 1
        sphten.SphericalTensor(torch.randn(mul * (lmax + 1) ** 2), lmax)
        mul = 3
        sphten.FourierTensor(torch.randn(mul * (lmax + 1) ** 2), mul, lmax)

    def test_from_geometry(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sphten.SphericalTensor.from_geometry(coords, lmax)

    def test_from_geometry_with_radial(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]

        def radial_model(x):
            return torch.ones_like(x).unsqueeze(-1)

        sphten.FourierTensor.from_geometry(coords, radial_model, lmax)

    def test_sph_norm(self):
        torch.set_default_dtype(torch.float64)
        lmax = 6
        mul = 1
        sph = sphten.SphericalTensor(torch.randn(mul * (lmax + 1) ** 2), lmax)
        sph.sph_norm()

        mul = 3
        sph = sphten.FourierTensor(torch.randn(
            mul * (lmax + 1) ** 2), mul, lmax)
        sph.sph_norm()

    def test_plot(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]
        sph = sphten.SphericalTensor.from_geometry(coords, lmax)

        n = 16
        r, f = sph.plot(n=n)
        assert r.shape[2] == 3
        assert f.shape[:2] == r.shape[:2]

    def test_plot_with_radial(self):
        torch.set_default_dtype(torch.float64)
        N = 4
        lmax = 6
        coords = torch.randn(N, 3)
        coords = coords[coords.norm(2, -1) > 0]

        def radial_model(x):
            return torch.ones_like(x).unsqueeze(-1)

        sph = sphten.FourierTensor.from_geometry(coords,
                                                 radial_model,
                                                 lmax)

        n = 16
        center = torch.ones(3)
        r, f = sph.plot(box_length=3.0, n=n, center=center)
        assert list(r.shape) == [n ** 3, 3]
        assert list(f.shape) == [n ** 3]

    def test_signal_on_sphere(self):
        torch.set_default_dtype(torch.float64)
        lmax = 4
        sph = sphten.SphericalTensor(torch.randn((lmax + 1)**2), lmax)

        r, val1 = sph.signal_on_grid(2 * (lmax + 1))
        val2 = sph.signal_xyz(r)
        assert (val1 - val2).abs().max() < 1e-10

    def test_change_lmax(self):
        lmax = 0
        mul = 1
        signal = torch.zeros(rs.dim([(mul, lmax)]))
        sph = sphten.SphericalTensor(signal, lmax)
        lmax_new = 5
        sph_new = sph.change_lmax(lmax_new)
        assert sph_new.signal.shape[0] == rs.dim(sph_new.Rs)

    def test_add(self):
        lmax = 4
        mul = 1
        signal1 = torch.zeros((lmax + 1) ** 2)
        signal2 = signal1.clone()
        signal1[0] = 1.
        signal2[3] = 1.
        sph1 = sphten.SphericalTensor(signal1, lmax)
        sph2 = sphten.SphericalTensor(signal2, lmax)

        new_sph = sph1 + sph2
        assert new_sph.mul == mul
        assert new_sph.lmax == max(sph1.lmax, sph2.lmax)

    def test_mul_and_dot(self):
        lmax = 4
        signal1 = torch.zeros((lmax + 1) ** 2)
        signal2 = signal1.clone()
        signal1[0] = 1.
        signal2[3] = 1.
        sph1 = sphten.SphericalTensor(signal1, lmax)
        sph2 = sphten.SphericalTensor(signal2, lmax)

        new_sph = sph1 * sph2
        assert rs.are_equal(new_sph.Rs, [(rs.mul_dim(sph1.Rs), 0, 0)])

        sph1.dot(sph2)


if __name__ == '__main__':
    unittest.main()
