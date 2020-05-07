# pylint: disable=not-callable, no-member, invalid-name, line-too-long
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math
from functools import lru_cache
from typing import List

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn.util.eval_code import eval_code


def spherical_harmonics_expand_matrix(ls, like=None):
    """
    convertion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    :return: tensor [l, m, l * m]
    """
    lmax = max(ls)
    zeros = torch.zeros if like is None else like.new_zeros
    m = zeros(len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls))
    i = 0
    for j, l in enumerate(ls):
        m[j, lmax - l: lmax + l + 1, i:i + 2 * l + 1] = torch.eye(2 * l + 1)
        i += 2 * l + 1
    return m


@torch.jit.script
def mul_m_lm(ls: List[int], x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
    """
    multiply tensor [..., l * m] by [..., m]
    """
    lmax = x_m.shape[-1] // 2
    out = []
    i = 0
    for l in ls:
        out.append(x_lm[..., i: i + 2 * l + 1] * x_m[..., lmax - l: lmax + l + 1])
        i += 2 * l + 1
    return torch.cat(out, dim=-1)


def sympy_legendre(l, m):
    """
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols('z y', real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1)**l, z, l + m)
    ex *= (-1)**l * sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


def poly_legendre(l, m):
    """
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols('z y', real=True)
    return Poly(sympy_legendre(l, m), domain='R', gens=(z, y)).as_dict()


_legendre_code = """
import torch
from e3nn import rsh

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
            p = poly_legendre(l, m)
            formula = " + ".join("{:.25f} * z**{} * y**{}".format(c, zn, yn) for (zn, yn), c in p.items())
            fill += "    l{} = {}\n".format(m, formula)

        for m in range(-l, l + 1):
            fill += "    out[..., {}] = l{}\n".format(i, abs(m))
            i += 1

    code = _legendre_code
    code = code.replace("lsize", str(sum(2 * l + 1 for l in ls)))
    code = code.replace("# fill out", fill)
    return eval_code(code).main


def legendre(ls, z, y=None):
    """
    associated Legendre polynomials

    :param ls: list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    if y is None:
        y = (1 - z**2).relu().sqrt()

    return _legendre_genjit(tuple(ls))(z, y)


def spherical_harmonics_z(ls, z, y=None):
    """
    the z componant of the spherical harmonics
    (useful to perform fourier transform)

    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    return legendre(ls, z, y)  # [..., l * m]


@torch.jit.script
def spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:
    """
    the alpha (x, y) componant of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
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
    ], dim=-1)

    return out  # [..., m]


def spherical_harmonics_alpha_beta(ls, alpha, beta):
    """
    spherical harmonics

    :param ls: list of int
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    if alpha.device.type == 'cuda' and beta.device.type == 'cuda' and not alpha.requires_grad and not beta.requires_grad and max(ls) <= 10:
        xyz = torch.stack([beta.sin() * alpha.cos(), beta.sin() * alpha.sin(), beta.cos()], dim=-1)
        try:
            return spherical_harmonics_xyz_cuda(ls, xyz)
        except ImportError:
            pass

    return spherical_harmonics_alpha_z_y(ls, alpha, beta.cos(), beta.sin().abs())


def spherical_harmonics_alpha_z_y(ls, alpha, z, y):
    """
    cpu version of spherical_harmonics_alpha_beta
    """
    sha = spherical_harmonics_alpha(max(ls), alpha.flatten())  # [z, m]
    shz = spherical_harmonics_z(ls, z.flatten(), y.flatten())  # [z, l * m]
    out = mul_m_lm(ls, sha, shz)
    return out.reshape(alpha.shape + (shz.shape[1],))


def spherical_harmonics_xyz(ls, xyz):
    """
    spherical harmonics

    :param ls: list of int
    :param xyz: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """

    if xyz.device.type == 'cuda' and not xyz.requires_grad and max(ls) <= 10:
        try:
            return spherical_harmonics_xyz_cuda(ls, xyz)
        except ImportError:
            pass

    xyz = xyz / torch.norm(xyz, 2, dim=-1, keepdim=True)

    alpha = torch.atan2(xyz[..., 1], xyz[..., 0])  # [...]
    z = xyz[..., 2]  # [...]
    y = (xyz[..., 0].pow(2) + xyz[..., 1].pow(2)).sqrt()  # [...]

    return spherical_harmonics_alpha_z_y(ls, alpha, z, y)


def spherical_harmonics_xyz_cuda(ls, xyz):
    """
    cuda version of spherical_harmonics_xyz
    """
    from e3nn import cuda_rsh  # pylint: disable=no-name-in-module, import-outside-toplevel

    *size, _ = xyz.size()
    xyz = xyz.reshape(-1, 3)
    xyz = xyz / torch.norm(xyz, 2, -1, keepdim=True)

    lmax = max(ls)
    out = xyz.new_empty(((lmax + 1)**2, xyz.size(0)))  # [ filters, batch_size]
    cuda_rsh.real_spherical_harmonics(out, xyz)

    # (-1)^L same as (pi-theta) -> (-1)^(L+m) and 'quantum' norm (-1)^m combined  # h - halved
    norm_coef = [elem for lh in range((lmax + 1) // 2) for elem in [1.] * (4 * lh + 1) + [-1.] * (4 * lh + 3)]
    if lmax % 2 == 0:
        norm_coef.extend([1.] * (2 * lmax + 1))
    norm_coef = out.new_tensor(norm_coef).unsqueeze(1)
    out.mul_(norm_coef)

    if ls != list(range(lmax + 1)):
        out = torch.cat([out[l**2: (l + 1)**2] for l in ls])
    return out.T.reshape(*size, -1)
