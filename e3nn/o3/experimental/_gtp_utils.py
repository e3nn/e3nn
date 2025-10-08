# Adapting https://github.com/atomicarchitects/PriceofFreedom/blob/main/src/tensor_products/gaunt_tensor_product_utils.py

import functools
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable


class RectangularSignal:
    """A signal defined on a rectangular region as a function of theta and phi."""

    def __init__(self, grid_values: torch.Tensor, res_theta: int, res_phi: int):
        self.grid_values = grid_values
        if res_theta <= 2 or res_phi <= 2:
            raise ValueError("res_theta and res_phi must be greater than 2.")

        self.res_theta = res_theta
        self.res_phi = res_phi
        assert self.grid_values.shape == (*self.grid_values.shape[:-2], res_theta, res_phi)

    @staticmethod
    def from_function(
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, res_theta: int, res_phi: int, wrap_theta: bool
    ):
        """Create a signal from a function of theta and phi."""
        if wrap_theta:

            def f_wrapped(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
                f_val = f(theta, phi)
                fd_val = -f(theta, phi)
                return torch.where(theta < torch.pi, f_val, fd_val)

            func = f_wrapped
        else:
            func = f
        thetas, phis = torch.meshgrid(RectangularSignal._thetas(res_theta), RectangularSignal._phis(res_phi), indexing="ij")
        grid_values = func(thetas, phis)

        return RectangularSignal(grid_values, res_theta, res_phi)

    def thetas(self):
        return RectangularSignal._thetas(self.res_theta)

    def phis(self):
        return RectangularSignal._phis(self.res_phi)

    @staticmethod
    def _thetas(res_theta: int):
        """Returns the theta values of the grid."""
        return torch.linspace(0, 2 * torch.pi, res_theta)

    @staticmethod
    def _phis(res_phi: int):
        """Returns the phi values of the grid."""
        return torch.linspace(0, 2 * torch.pi, res_phi)

    def integrate(self, area_element: str) -> torch.Tensor:
        """Computes the integral of the signal over a rectangular/spherical region."""
        if area_element == "rectangular":
            return self.integrate_rectangular()
        elif area_element == "spherical":
            return self.integrate_spherical()
        else:
            raise ValueError(f"Unknown area element {area_element}")

    def integrate_rectangular(self) -> torch.Tensor:
        """Computes the integral of the signal over the rectangular region."""
        return RectangularSignal._integrate(self.grid_values, self.thetas(), self.phis())

    def integrate_spherical(self) -> torch.Tensor:
        """Computes the integral of the signal over the spherical region."""
        # Only integrate upto theta = pi.
        thetas = self.thetas()[: self.res_theta // 2]
        grid_values = self.grid_values[..., : self.res_theta // 2, :]
        return RectangularSignal._integrate(grid_values * torch.sin(thetas)[:, None], thetas, self.phis())

    @staticmethod
    def _integrate(grid_values: torch.Tensor, thetas: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """Computes the integral of the signal over the rectangular region."""
        assert grid_values.shape == (len(thetas), len(phis))

        dtheta = thetas[1] - thetas[0]
        theta_weights = torch.cat([torch.tensor([0.5]), torch.ones(len(thetas) - 2), torch.tensor([0.5])])
        integral = torch.sum(grid_values * theta_weights[:, None], dim=0) * dtheta
        assert integral.shape == (len(phis),)

        dphi = phis[1] - phis[0]
        phi_weights = torch.cat([torch.tensor([0.5]), torch.ones(len(phis) - 2), torch.tensor([0.5])])
        integral = torch.sum(integral * phi_weights, dim=0) * dphi
        assert integral.shape == ()
        return integral

    def __mul__(self, other):
        """Pointwise multiplication of two signals."""
        assert isinstance(other, RectangularSignal)
        assert self.res_theta == other.res_theta
        assert self.res_phi == other.res_phi
        return RectangularSignal(self.grid_values * other.grid_values, self.res_theta, self.res_phi)


def from_lm_index(lm_index: int) -> tuple:
    """Converts a grid index to l, m values."""
    l = torch.floor(torch.sqrt(torch.tensor(lm_index, dtype=torch.float))).int()
    m = lm_index - l * (l + 1)
    return l.item(), m.item()


def to_lm_index(l: int, m: int) -> int:
    """Converts l, m values to a grid index."""
    return l * (l + 1) + m


def sh_phi(l: int, m: int, phi: torch.Tensor) -> torch.Tensor:
    """Phi dependence of spherical harmonics."""
    assert phi.dim() == 0
    phi = phi[..., None]  # [..., 1]
    ms = torch.arange(1, l + 1)  # [1, 2, 3, ..., l]
    cos = torch.cos(ms * phi)  # [..., m]

    ms = torch.arange(l, 0, -1)  # [l, l-1, l-2, ..., 1]
    sin = torch.sin(ms * phi)  # [..., m]

    return torch.cat(
        [
            torch.sqrt(torch.tensor(2.0)) * sin,
            torch.ones_like(phi),
            torch.sqrt(torch.tensor(2.0)) * cos,
        ],
        dim=-1,
    )[l + m]


def sh_theta(l: int, m: int, theta: torch.Tensor) -> torch.Tensor:
    """Theta dependence of spherical harmonics."""
    assert theta.dim() == 0
    cos_theta = torch.cos(theta)
    # Note: This is a simplification. You may need to implement a more accurate Legendre polynomial calculation.
    legendres = torch.special.legendre_polynomial(l, cos_theta)
    sh_theta_comp = legendres[abs(m)]
    return sh_theta_comp


def spherical_harmonic(l: int, m: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Spherical harmonic (Y_lm)"""

    def Y_lm(theta, phi):
        assert theta.shape == phi.shape
        return sh_theta(l, m, theta) * sh_phi(l, m, phi)

    return Y_lm


def fourier_2D(u: int, v: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Fourier function in 2D."""

    def fourier_uv(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * (u * theta + v * phi)) / (2 * torch.pi)

    return fourier_uv


@functools.lru_cache(maxsize=None)
def create_spherical_harmonic_signal(l: int, m: int, *, res_theta: int, res_phi: int, wrap_theta: bool):
    """Creates a signal for Y^{l,m}."""
    return RectangularSignal.from_function(
        spherical_harmonic(l, m),
        res_theta=res_theta,
        res_phi=res_phi,
        wrap_theta=wrap_theta,
    )


@functools.lru_cache(maxsize=None)
def create_fourier_2D_signal(u: int, v: int, *, res_theta: int, res_phi: int, wrap_theta: bool):
    """Creates a signal for Fourier function defined by {u, v}."""
    return RectangularSignal.from_function(
        fourier_2D(u, v),
        res_theta=res_theta,
        res_phi=res_phi,
        wrap_theta=wrap_theta,
    )


def to_u_index(u: int, lmax: int) -> int:
    """Returns the index of u in the grid."""
    return u + lmax


def to_v_index(v: int, lmax: int) -> int:
    """Returns the index of v in the grid."""
    return v + lmax


@functools.lru_cache(maxsize=None)
def compute_y(l: int, m: int, u: int, v: int, *, res_theta: int, res_phi: int):
    """Computes y^{l,m}_{u, v}."""
    Y_signal = create_spherical_harmonic_signal(l, m, res_theta=res_theta, res_phi=res_phi, wrap_theta=(m % 2))
    F_signal = create_fourier_2D_signal(u, v, res_theta=res_theta, res_phi=res_phi, wrap_theta=False)
    return (Y_signal * F_signal).integrate(area_element="rectangular")


@functools.lru_cache(maxsize=None)
def compute_y_grid(lmax: int, *, res_theta: int, res_phi: int):
    """Computes the grid of y^{l,m}_{u, v}."""
    lm_indices = torch.arange((lmax + 1) ** 2)
    us = torch.arange(-lmax, lmax + 1)
    vs = torch.arange(-lmax, lmax + 1)
    mesh = torch.meshgrid(lm_indices, us, vs, indexing="ij")

    y_grid = torch.zeros(((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1), dtype=torch.complex64)
    for lm_index, u, v in zip(*[m.ravel() for m in mesh]):
        l, m = from_lm_index(lm_index.item())
        u_index = to_u_index(u.item(), lmax)
        v_index = to_v_index(v.item(), lmax)
        y_val = compute_y(l, m, u.item(), v.item(), res_theta=res_theta, res_phi=res_phi)
        y_grid[lm_index, u_index, v_index] = y_val

    assert y_grid.shape == ((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1)
    return y_grid


@functools.lru_cache(maxsize=None)
def compute_z(l: int, m: int, u: int, v: int, *, res_theta: int, res_phi: int):
    """Computes z^{l,m}_{u, v}."""
    Y_signal = create_spherical_harmonic_signal(l, m, res_theta=res_theta, res_phi=res_phi, wrap_theta=(m % 2))
    F_signal = create_fourier_2D_signal(u, v, res_theta=res_theta, res_phi=res_phi, wrap_theta=False)
    return (Y_signal * F_signal).integrate(area_element="spherical")


@functools.lru_cache(maxsize=None)
def compute_z_grid(lmax: int, *, res_theta: int, res_phi: int):
    """Computes the grid of z^{l,m}_{u, v}."""
    lm_indices = torch.arange((lmax + 1) ** 2)
    us = torch.arange(-lmax, lmax + 1)
    vs = torch.arange(-lmax, lmax + 1)
    mesh = torch.meshgrid(lm_indices, us, vs, indexing="ij")

    z_grid = torch.zeros(((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1), dtype=torch.complex64)
    for lm_index, u, v in zip(*[m.ravel() for m in mesh]):
        l, m = from_lm_index(lm_index.item())
        u_index = to_u_index(u.item(), lmax)
        v_index = to_v_index(v.item(), lmax)
        z_val = compute_z(l, m, u.item(), v.item(), res_theta=res_theta, res_phi=res_phi)
        z_grid[lm_index, u_index, v_index] = z_val

    assert z_grid.shape == ((lmax + 1) ** 2, 2 * lmax + 1, 2 * lmax + 1)
    return z_grid


def convolve_2D_fft(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """2D convolution of x1 and x2 using FFT."""
    return _convolve_2D_fft_single_sample(x1, x2)


def _convolve_2D_fft_single_sample(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """2D convolution of x1 and x2 using FFT for a single sample."""
    assert x1.dim() == x2.dim() == 2

    # Get dimensions.
    x1_dim1, x1_dim2 = x1.shape
    x2_dim1, x2_dim2 = x2.shape

    # Calculate full output size.
    full_dim1 = x1_dim1 + x2_dim1 - 1
    full_dim2 = x1_dim2 + x2_dim2 - 1

    # Pad x1 and x2.
    x1_padded = F.pad(x1, (0, full_dim2 - x1_dim2, 0, full_dim1 - x1_dim1))
    x2_padded = F.pad(x2, (0, full_dim2 - x2_dim2, 0, full_dim1 - x2_dim1))

    # Perform FFT.
    x1_fft = torch.fft.fft2(x1_padded)
    x2_fft = torch.fft.fft2(x2_padded)

    # Multiply in frequency domain.
    result_fft = x1_fft * x2_fft

    # Inverse FFT.
    result = torch.fft.ifft2(result_fft)

    return result


def convolve_2D_direct(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """2D convolution of x1 and x2 directly."""

    def convolve_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (
            F.conv2d(
                x1.unsqueeze(0).unsqueeze(0),
                x2.flip(-1, -2).unsqueeze(0).unsqueeze(0),
                padding=(x2.shape[0] - 1, x2.shape[1] - 1),
            )
            .squeeze(0)
            .squeeze(0)
        )

    # Handle additional dimensions
    for _ in range(x1.dim() - 2):
        convolve_fn = torch.vmap(convolve_fn)

    return convolve_fn(x1, x2)
