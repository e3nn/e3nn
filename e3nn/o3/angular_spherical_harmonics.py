r"""Spherical Harmonics as functions of Euler angles
"""
import math
from functools import lru_cache
from typing import List, Tuple

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn.util import eval_code
from e3nn import o3


def spherical_harmonics_alpha_beta(l, alpha, beta):
    r"""Spherical harmonics of :math:`\vec r = R_y(\alpha) R_x(\beta) e_y`

    .. math:: Y^l(\alpha, \beta) = S^l(\alpha) P^l(\cos(\beta))

    where :math:`P^l` are the `legendre` polynomials


    Parameters
    ----------
    l : int or list of int
        degree of the spherical harmonics.

    alpha : `torch.Tensor`
        tensor of shape ``(...)``.

    beta : `torch.Tensor`
        tensor of shape ``(...)``.

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``
    """
    if isinstance(l, o3.Irreps):
        ls = [l for mul, (l, p) in l for _ in range(mul)]
    elif isinstance(l, int):
        ls = [l]
    else:
        ls = list(l)

    y, z = beta.cos(), beta.sin()
    sha = spherical_harmonics_alpha(max(ls), alpha.flatten())  # [z, m]
    shy = legendre(ls, y.flatten(), z.flatten())  # [z, l * m]
    out = _mul_m_lm([(1, l) for l in ls], sha, shy)
    return out.reshape(alpha.shape + (shy.shape[1],))


@torch.jit.script
def spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:
    r""":math:`S^l(\alpha)` of `spherical_harmonics_alpha_beta`

    Parameters
    ----------
    l : int
        degree of the spherical harmonics.

    alpha : `torch.Tensor`
        tensor of shape ``(...)``.

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``
    """
    alpha = alpha.unsqueeze(-1)  # [..., 1]
    m = torch.arange(1, l + 1, dtype=alpha.dtype, device=alpha.device)  # [1, 2, 3, ..., l]
    cos = torch.cos(m * alpha)  # [..., m]

    m = torch.arange(l, 0, -1, dtype=alpha.dtype, device=alpha.device)  # [l, l-1, l-2, ..., 1]
    sin = torch.sin(m * alpha)  # [..., m]

    out = torch.cat([
        math.sqrt(2) * sin,
        torch.ones_like(alpha),
        math.sqrt(2) * cos,
    ], dim=alpha.ndim-1)

    return out  # [..., m]


def legendre(l, z, y=None):
    r"""Legendre polynomials

    Parameters
    ----------
    l : int or list of int
        degree of the polynomial.

    y : `torch.Tensor`
        tensor of shape ``(...)``.

    y : `torch.Tensor`, optional
        tensor of shape ``(...)``.

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``
    """
    if isinstance(l, o3.Irreps):
        ls = [l for mul, (l, p) in l for _ in range(mul)]
    elif isinstance(l, int):
        ls = [l]
    else:
        ls = list(l)

    if y is None:
        y = (1 - z**2).relu().sqrt()

    return _legendre_genjit(tuple(ls))(z, y)


_legendre_code = """
import torch

@torch.jit.script
def main(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = z.new_zeros(z.shape + (lsize,))

# fill out

    return out
"""


@lru_cache()
def _legendre_genjit(ls):
    ls = list(ls)
    fill = ""
    i = 0
    for l in ls:
        for m in range(l + 1):
            p = _poly_legendre(l, m)
            formula = " + ".join("{:.25f} * z**{} * y**{}".format(c, zn, yn) for (zn, yn), c in p.items())
            fill += "    l{} = {}\n".format(m, formula)

        for m in range(-l, l + 1):
            fill += "    out[..., {}] = l{}\n".format(i, abs(m))
            i += 1

    code = _legendre_code
    code = code.replace("lsize", str(sum(2 * l + 1 for l in ls)))
    code = code.replace("# fill out", fill)
    return eval_code(code).main


def _poly_legendre(l, m):
    r"""
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols('z y', real=True)
    return Poly(_sympy_legendre(l, m), domain='R', gens=(z, y)).as_dict()


def _sympy_legendre(l, m):
    r"""
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)
    - remove (-1)^l

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols('z y', real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1)**l, z, l + m)
    ex *= sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


@torch.jit.script
def _mul_m_lm(mul_l: List[Tuple[int, int]], x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
    """
    multiply tensor [..., l * m] by [..., m]
    """
    l_max = x_m.shape[-1] // 2
    out = []
    i = 0
    for mul, l in mul_l:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i: i + d]  # [..., mul * m]
        x1 = x1.reshape(x1.shape[:-1] + (mul, 2 * l + 1))  # [..., mul, m]
        x2 = x_m[..., l_max - l: l_max + l + 1]  # [..., m]
        x2 = x2.reshape(x2.shape[:-1] + (1, 2 * l + 1))  # [..., mul=1, m]
        x = x1 * x2
        x = x.reshape(x.shape[:-2] + (d,))
        out.append(x)
        i += d
    return torch.cat(out, dim=-1)
