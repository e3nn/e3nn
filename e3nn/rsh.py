# pylint: disable=not-callable, no-member, invalid-name, line-too-long
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math
import os

import torch
from sympy import Integer, Poly, diff, factorial, sqrt, symbols, pi

from e3nn.util.cache_file import cached_picklesjar


def spherical_harmonics_expand_matrix(lmax):
    """
    convertion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    :return: tensor [l, m, l * m]
    """
    m = torch.zeros(lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in range(lmax + 1)))
    i = 0
    for l in range(lmax + 1):
        m[l, lmax - l: lmax + l + 1, i:i + 2 * l + 1] = torch.eye(2 * l + 1)
        i += 2 * l + 1
    return m


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


@cached_picklesjar(os.path.join(os.path.dirname(__file__), 'cache/legendre'), maxsize=None)
def poly_legendre(l, m):
    """
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols('z y', real=True)
    p = Poly(sympy_legendre(l, m), domain='R', gens=(z, y))
    return {exp: float(coef) for exp, coef in p.as_dict().items()}


def legendre(ls, z, y=None):
    """
    associated Legendre polynomials

    :param ls: list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    if y is None:
        y = (1 - z**2).relu().sqrt()

    zs = [z**m for m in range(max(ls) + 1)]
    ys = [y**m for m in range(max(ls) + 1)]

    ps = []
    for l in ls:
        p = torch.stack([sum(coef * zs[nz] * ys[ny] for (nz, ny), coef in poly_legendre(l, abs(m)).items()) for m in range(-l, 1)], dim=-1)
        ps += [torch.cat([p, p[..., :-1].flip(-1)], dim=-1)]
    return torch.cat(ps, dim=-1)


def spherical_harmonics_beta(ls, cosbeta, abssinbeta=None):
    """
    the cosbeta componant of the spherical harmonics
    (useful to perform fourier transform)

    :param cosbeta: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    return legendre(ls, cosbeta, abssinbeta)  # [..., l * m]


def spherical_harmonics_alpha(l, alpha):
    """
    the alpha componant of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    size = alpha.shape
    alpha = alpha.reshape(-1, 1)  # [batch, 1]

    m = torch.arange(1, l + 1).flip(0).to(alpha)  # [l, l-1, l-2, ..., 1]
    sin = torch.sin(m * alpha)  # [batch, m]

    m = torch.arange(1, l + 1).to(alpha)  # [1, 2, 3, ..., l]
    cos = torch.cos(m * alpha)  # [batch, m]

    out = torch.cat([
        math.sqrt(2) * sin,
        torch.ones_like(alpha),
        math.sqrt(2) * cos,
    ], dim=-1)

    return out.reshape(*size, 2 * l + 1)  # [..., m]


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

    output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], beta.cos(), beta.sin().abs()) for l in ls]
    return torch.cat(output, dim=-1)


def spherical_harmonics_xyz(ls, xyz):
    """
    spherical harmonics

    :param ls: list of int
    :param xyz: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """
    norm = torch.norm(xyz, 2, -1, keepdim=True)
    xyz = xyz / norm

    if xyz.device.type == 'cuda' and not xyz.requires_grad and max(ls) <= 10:
        try:
            return spherical_harmonics_xyz_cuda(ls, xyz)
        except ImportError:
            pass

    alpha = torch.atan2(xyz[..., 1], xyz[..., 0])  # [...]
    cosbeta = xyz[..., 2]  # [...]
    abssinbeta = (xyz[..., 1].pow(2) + xyz[..., 0].pow(2)).sqrt()
    output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], cosbeta, abssinbeta) for l in ls]
    return torch.cat(output, dim=-1)


def spherical_harmonics_xyz_cuda(ls, xyz):
    """
    cuda version of spherical_harmonics_xyz
    """
    from e3nn import cuda_rsh  # pylint: disable=no-name-in-module, import-outside-toplevel

    *size, _ = xyz.size()
    xyz = xyz.reshape(-1, 3)
    lmax = max(ls)
    out = xyz.new_empty(((lmax + 1)**2, xyz.size(0)))  # [ filters, batch_size]
    cuda_rsh.real_spherical_harmonics(out, xyz)

    # (-1)^L same as (pi-theta) -> (-1)^(L+m) and 'quantum' norm (-1)^m combined  # h - halved
    norm_coef = [elem for lh in range((lmax + 1) // 2) for elem in [1.] * (4 * lh + 1) + [-1.] * (4 * lh + 3)]
    if lmax % 2 == 0:
        norm_coef.extend([1.] * (2 * lmax + 1))
    norm_coef = torch.tensor(norm_coef).to(out).unsqueeze(1)
    out.mul_(norm_coef)

    if ls != list(range(lmax + 1)):
        out = torch.cat([out[l**2: (l + 1)**2] for l in ls])
    return out.T.reshape(*size, -1)
