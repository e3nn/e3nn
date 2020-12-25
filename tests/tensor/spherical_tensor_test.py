# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch

from e3nn import o3, rs
from e3nn.tensor.spherical_tensor import spherical_harmonics_dirac, SphericalTensor, projection, adjusted_projection
from e3nn.tensor.fourier_tensor import FourierTensor
from e3nn.tensor.irrep_tensor import IrrepTensor


def test_sh_dirac():
    with o3.torch_default_dtype(torch.float64):
        for l in range(5):
            r = torch.randn(3)
            a = spherical_harmonics_dirac(r, l)
            v = SphericalTensor(a).signal_xyz(r)
            assert v.sub(1).abs() < 1e-10


def test_projection():
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]
    projection(coords, lmax)


def test_adjusted_projection():
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]
    adjusted_projection(coords, lmax)


def test_SphericalTensor():
    torch.set_default_dtype(torch.float64)
    lmax = 6
    SphericalTensor(torch.randn((lmax + 1) ** 2))
    mul = 3
    FourierTensor(torch.randn(mul * (lmax + 1) ** 2), mul, lmax)


def test_from_geometry():
    torch.set_default_dtype(torch.float64)
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]
    SphericalTensor.from_geometry(coords, lmax)


def test_from_samples():
    torch.set_default_dtype(torch.float64)
    lmax = 2
    signal1 = torch.randn((lmax + 1)**2)
    r, v = SphericalTensor(signal1).signal_on_grid(60)
    signal2 = SphericalTensor.from_samples(r, v, res=200, lmax=lmax).signal
    assert (signal1 - signal2).abs().max() < 0.01


def test_from_geometry_with_radial():
    torch.set_default_dtype(torch.float64)
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]

    def radial_model(x):
        return torch.ones_like(x).unsqueeze(-1)

    FourierTensor.from_geometry(coords, radial_model, lmax)


def test_sph_norm():
    torch.set_default_dtype(torch.float64)
    lmax = 6
    sph = SphericalTensor(torch.randn((lmax + 1) ** 2))
    sph.sph_norm()


def test_plot():
    torch.set_default_dtype(torch.float64)
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]
    sph = SphericalTensor.from_geometry(coords, lmax)

    n = 16
    r, f = sph.plot(res=n)
    assert r.shape[2] == 3
    assert f.shape[:2] == r.shape[:2]


def test_plot_with_radial():
    torch.set_default_dtype(torch.float64)
    N = 4
    lmax = 6
    coords = torch.randn(N, 3)
    coords = coords[coords.norm(2, -1) > 0]

    def radial_model(x):
        return torch.ones_like(x).unsqueeze(-1)

    sph = FourierTensor.from_geometry(coords, radial_model, lmax)

    n = 16
    center = torch.ones(3)
    r, f = sph.plot(box_length=3.0, n=n, center=center)
    assert list(r.shape) == [n ** 3, 3]
    assert list(f.shape) == [n ** 3]


def test_signal_on_sphere():
    torch.set_default_dtype(torch.float64)
    lmax = 4
    sph = SphericalTensor(torch.randn((lmax + 1)**2))

    r, val1 = sph.signal_on_grid(2 * (lmax + 1))
    val2 = sph.signal_xyz(r)
    assert (val1 - val2).abs().max() < 1e-10


def test_change_lmax():
    sph = SphericalTensor(torch.zeros(1))
    sph_new = sph.change_lmax(5)
    assert sph_new.signal.shape[0] == rs.dim(sph_new.Rs)


def test_add():
    lmax = 4
    signal1 = torch.zeros((lmax + 1) ** 2)
    signal2 = signal1.clone()
    signal1[0] = 1.
    signal2[3] = 1.
    sph1 = SphericalTensor(signal1)
    sph2 = SphericalTensor(signal2)

    new_sph = sph1 + sph2
    assert new_sph.lmax == max(sph1.lmax, sph2.lmax)


def test_mul_and_dot():
    lmax = 4
    signal1 = torch.zeros((lmax + 1) ** 2)
    signal2 = signal1.clone()
    signal1[0] = 1.
    signal2[3] = 1.
    sph1 = SphericalTensor(signal1)
    sph2 = SphericalTensor(signal2)

    new_sph = sph1 * sph2
    assert rs.are_equal(new_sph.Rs, [(rs.mul_dim(sph1.Rs), 0, 0)])

    sph1.dot(sph2)


def test_from_irrep_tensor():
    irrep = IrrepTensor(torch.randn(6), Rs=[(2, 1, 0)])
    try:
        SphericalTensor.from_irrep_tensor(irrep)
    except:
        pass  # Exception was raised
    else:
        raise AssertionError("ValueError was not raised.")

    irrep = IrrepTensor(torch.randn(6), Rs=[(1, 1, 1), (1, 1, -1)])
    try:
        SphericalTensor.from_irrep_tensor(irrep)
    except:
        pass  # Exception was raised
    else:
        raise AssertionError("ValueError was not raised.")

    irrep = IrrepTensor(torch.ones(8), Rs=[(1, 0, 0), (1, 3, 0)])
    sph = SphericalTensor.from_irrep_tensor(irrep)
    compare = torch.zeros(16)
    compare[0] = 1.
    compare[-7:] = 1.
    assert torch.allclose(sph.signal, compare)
