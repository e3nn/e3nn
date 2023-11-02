"""This module provides functions for computing spherical harmonics and associated
Legendre polynomials on 3D points using PyTorch. It leverages caching and JIT compilation
for efficient batch processing on CPU or GPU devices.

The main functions include:
- `calc_Ylm`: Computes spherical harmonics up to a given degree for a set of points.
- `calc_Plm`: Calculates associated Legendre polynomials for cos(theta) values.

Utilizing memoization, intermediate results such as 'A' and 'B' coefficients and factorials
are stored for repeated use. The JIT-compiled functions offer optimized computation
for these mathematical constructs essential in fields such as quantum mechanics,
electromagnetics, and computer graphics.
"""

from functools import cache
import math
from typing import Tuple

import torch


@cache
def get_klm(max_deg: int, device: str = "cpu", dtype=torch.long) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate and cache the values of k, l, and m for the spherical harmonics calculation.

    Args:
        max_deg (int): The maximum degree for the spherical harmonics.
        device (str, optional): The device on which to perform calculations. Default is 'cpu'.
        dtype (torch.dtype, optional): The data type for the tensors. Default is torch.long.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors representing k, l, m, power, and mask.
    """
    k = torch.arange(max_deg + 1, device=device, dtype=dtype).unsqueeze(0)
    lm_list = [[d, m] for d in range(max_deg + 1) for m in range(-d, d + 1)]
    l, m = torch.tensor(lm_list, device=device, dtype=dtype).T.reshape(2, -1, 1)
    mask = m >= 0
    m = torch.abs(m)
    power = 2 * k - l - m
    power[power < 0] = 0  # When "power"<0 the coefficient is 0
    return k, l, m, power, mask


@cache
def get_factorial(N, nan=False, device: str = "cpu", dtype=torch.float64):
    """Calculate the factorial of all integers up to N and cache the result.

    Args:
        N (int): The maximum number to calculate the factorial for.
        nan (bool, optional): If True, append NaN to the end of the factorial tensor. Default is False.
        device (str, optional): The device on which to perform calculations. Default is 'cpu'.
        dtype (torch.dtype, optional): The data type for the tensor. Default is torch.float64.

    Returns:
        torch.Tensor: A tensor of factorials from 0 to N, with optional NaN appended.
    """
    ints = torch.arange(N + 1, device=device, dtype=dtype)
    ints[0] = 1  # Make sure that factorial(0) = 1
    factorial = torch.cumprod(ints, dim=0)
    if not nan:
        return factorial
    nan_tensor = torch.full_like(factorial, float("nan"))
    return torch.cat((factorial, nan_tensor))


@cache
def get_A(max_degree: int, device: str = "cpu", dtype=torch.float64) -> torch.tensor:
    """Calculate and cache the 'A' coefficients used in the spherical harmonics and
    associated Legendre polynomials calculations.

    Args:
        max_degree (int): The maximum degree for the spherical harmonics and associated
                          Legendre polynomials.
        device (str, optional): The device on which to perform calculations. Default is 'cpu'.
        dtype (torch.dtype, optional): The data type for the tensors. Default is torch.float64.

    Returns:
        torch.Tensor: A tensor representing the 'A' coefficients for both spherical harmonics
                      and associated Legendre polynomials.
    """
    # Retrieve cached values
    k, l, m, _, _ = get_klm(max_degree, device=device, dtype=torch.long)  # Use abs_m
    factorial = get_factorial(2 * max_degree, nan=True, device=device, dtype=dtype)  # dtype!=long for nan=True

    # Calculate
    A = (
        (-1) ** (m + l - k) / 2**l * factorial[2 * k] / factorial[k] / factorial[l - k] / factorial[2 * k - l - m]
    )  # Don't use the cached power to exploit the locaiton of the nan values
    A[torch.isnan(A)] = 0.0
    return A


@torch.jit.script
def calc_Plm_jit(x: torch.Tensor, A: torch.Tensor, m: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """JIT-compiled function to calculate the associated Legendre polynomials using
    the precomputed 'A' coefficients.

    Args:
        x (torch.Tensor): Input tensor containing the cos(theta) values,
                          where theta is the polar angle.
        A (torch.Tensor): Precomputed 'A' coefficients tensor.
        m (torch.Tensor): Precomputed 'm' values tensor related to the azimuthal dependency.
        power (torch.Tensor): Precomputed power tensor for raising cos(theta) to various powers.

    Returns:
        torch.Tensor: The computed associated Legendre polynomials for the input tensor 'x'.
    """
    # Calculate
    x = x.unsqueeze(1)
    temp = torch.pow(x.unsqueeze(0), power.unsqueeze(1))
    pre_Plm = (temp @ A.unsqueeze(2)).squeeze(2).T
    Plm = pre_Plm * (1 - x**2) ** (m.T / 2)
    return Plm


def calc_Plm(max_degree: int, x: torch.Tensor) -> torch.Tensor:
    """Compute the associated Legendre polynomials of degree up to `max_degree` for
    a batch of cos(theta) values `x`.

    Args:
        max_degree (int): The maximum degree for the associated Legendre polynomials calculation.
        x (torch.Tensor): The input tensor containing the cos(theta) values for which to
                          calculate the associated Legendre polynomials.

    Returns:
        torch.Tensor: The computed associated Legendre polynomials values for each value in 'x'.
    """
    # Retrieve cached values
    device = x.device
    dtype = x.dtype
    A = get_A(max_degree, device=device, dtype=dtype)
    _, _, m, power, _ = get_klm(max_degree, device=device, dtype=torch.long)
    Plm = calc_Plm_jit(x, A, m, power)
    return Plm

@cache
def get_B(max_degree, device: str = "cpu", dtype=torch.float64):
    """Calculate and cache the 'B' coefficients used in the spherical harmonics calculation.

    Args:
        max_degree (int): The maximum degree for the spherical harmonics.
        device (str, optional): The device on which to perform calculations. Default is 'cpu'.
        dtype (torch.dtype, optional): The data type for the tensors. Default is torch.float64.

    Returns:
        torch.Tensor: A tensor representing the 'B' coefficients.
    """
    # Retrieve cached values
    k, l, m, _, _ = get_klm(max_degree, device=device, dtype=torch.long)  # Use abs(m)
    factorial = get_factorial(2 * max_degree, device=device, dtype=dtype)
    A = get_A(max_degree, device=device, dtype=dtype)

    # Calculate
    B = A * (-1) ** m * 2 ** ((m != 0) / 2) * torch.sqrt((2 * l + 1) / (4 * math.pi) * factorial[l - m] / factorial[l + m])
    return B


@torch.jit.script
def calc_Ylm_jit(x: torch.Tensor, B: torch.Tensor, m: torch.Tensor, power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate the spherical harmonics using the jit compiled script for performance.

    Args:
        x (torch.Tensor): Input tensor containing the coordinates.
        B (torch.Tensor): Precomputed 'B' coefficients tensor.
        m (torch.Tensor): Precomputed 'm' values tensor.
        power (torch.Tensor): Precomputed power tensor for the spherical harmonics.
        mask (torch.Tensor): Precomputed mask tensor for positive m values.

    Returns:
        torch.Tensor: The computed spherical harmonics for the input tensor 'x'.
    """
    # Preliminary calculations
    r = torch.norm(x, dim=1, keepdim=True)
    cos_theta = x[..., 2:3] / r
    phi = torch.atan2(x[..., 1:2], x[..., 0:1])

    # Calcuate coeff * Associated Legndre Polynomial
    cos_pows = cos_theta ** power.unsqueeze(1)
    pre_Plm = (cos_pows @ B.unsqueeze(2)).squeeze(2).T
    sin_pows = torch.sqrt(1 - cos_theta**2) ** m.T
    Plm = pre_Plm * sin_pows

    # Do the cases
    m_phi = m.T * phi
    cos_values = torch.cos(m_phi)
    sin_values = torch.sin(m_phi)
    cases = torch.where(mask.T, cos_values, sin_values)
    return Plm * cases


def calc_Ylm_jit(x: torch.Tensor, B: torch.Tensor, m: torch.Tensor, power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Preliminary calculations
    r = torch.norm(x, dim=1, keepdim=True)
    cos_theta = x[..., 2:3] / r
    phi = torch.atan2(x[..., 1:2], x[..., 0:1])

    # Calcuate coeff * Associated Legndre Polynomial
    cos_pows = cos_theta ** power.unsqueeze(1)
    pre_Plm = (cos_pows @ B.unsqueeze(2)).squeeze(2).T
    sin_pows = torch.sqrt(1 - cos_theta**2) ** m.T
    Plm = pre_Plm * sin_pows

    # Do the cases
    m_phi = m.T * phi
    cos_values = torch.cos(m_phi)
    sin_values = torch.sin(m_phi)
    cases = torch.where(mask.T, cos_values, sin_values)
    return Plm * cases


def calc_Ylm(max_degree: int, x: torch.Tensor) -> torch.Tensor:
    """Compute the spherical harmonics of degree up to `max_degree` for a batch of points `x`.

    Args:
        max_degree (int): The maximum degree for the spherical harmonics calculation.
        x (torch.Tensor): The input tensor containing the 3D points for which to calculate the spherical harmonics.

    Returns:
        torch.Tensor: The computed spherical harmonics values for each point in 'x'.
    """
    # Retrieve cached values
    device = x.device
    dtype = x.dtype
    len_x = len(x)
    B = get_B(max_degree, device=device, dtype=dtype)
    _, _, m, power, mask = get_klm(max_degree, device=device, dtype=torch.long)

    # Calculate
    Ylm = calc_Ylm_jit(x, B, m, power, mask)
    return Ylm
