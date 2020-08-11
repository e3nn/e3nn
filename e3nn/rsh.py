# pylint: disable=not-callable, no-member, invalid-name, line-too-long
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math
from functools import lru_cache

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn import rs
from e3nn.rs import TY_RS_STRICT, TY_RS_LOOSE
from e3nn.util.eval_code import eval_code

try:
    from e3nn import real_spherical_harmonics
    rsh_no_cuda = False
except ImportError:
    rsh_no_cuda = True


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
def mul_m_lm(Rs: rs.TY_RS_STRICT, x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    multiply tensor [..., l * m] by [..., m]
    """
    lmax = x_m.shape[-1] // 2
    out = []
    i = 0
    for mul, l, _ in Rs:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i: i + d]  # [..., mul * m]
        x1 = x1.reshape(x1.shape[:-1] + (mul, 2 * l + 1))  # [..., mul, m]
        x2 = x_m[..., lmax - l: lmax + l + 1]  # [..., m]
        x2 = x2.reshape(x2.shape[:-1] + (1, 2 * l + 1))  # [..., mul=1, m]
        x = x1 * x2
        x = x.reshape(x.shape[:-2] + (d,))
        out.append(x)
        i += d
    return torch.cat(out, dim=-1)


@torch.jit.script
def mul_radial_angular(Rs: rs.TY_RS_STRICT, radial, angular):  # pragma: no cover
    """
    :param Rs: output representation
    :param angular: [..., l * m]
    :param radial: [..., l * mul]
    """
    n = 0
    for mul, l, _ in Rs:
        n += mul * (2 * l + 1)

    y = radial[..., 0] * angular[..., 0]
    out = radial.new_empty(y.shape + (n,))

    a = 0
    r = 0
    i = 0
    for mul, l, _ in Rs:
        dim = mul * (2 * l + 1)
        x = radial[..., r: r + mul, None] * angular[..., None, a: a + 2 * l + 1]
        x = x.reshape(y.shape + (dim,))
        out[..., i: i + dim] = x
        i += dim
        r += mul
        a += 2 * l + 1

    assert r == radial.shape[-1]
    assert a == angular.shape[-1]

    return out


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


def spherical_harmonics_z(Rs, z, y=None):
    """
    the z component of the spherical harmonics
    (useful to perform fourier transform)

    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    Rs = rs.simplify(Rs)
    for _, l, p in Rs:
        assert p in [0, (-1)**l]
    ls = [l for mul, l, _ in Rs]
    return legendre(ls, z, y)  # [..., l * m]


@torch.jit.script
def spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    the alpha (x, y) component of the spherical harmonics
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
    ], dim=alpha.ndim-1)

    return out  # [..., m]


def spherical_harmonics_alpha_z_y(Rs: TY_RS_STRICT, alpha, z, y):
    """
    cpu version of spherical_harmonics_alpha_beta
    """
    sha = spherical_harmonics_alpha(rs.lmax(Rs), alpha.flatten())  # [z, m]
    shz = spherical_harmonics_z(Rs, z.flatten(), y.flatten())  # [z, l * m]
    out = mul_m_lm(Rs, sha, shz)
    return out.reshape(alpha.shape + (shz.shape[1],))


def spherical_harmonics_xyz_cuda(Rs: TY_RS_STRICT, xyz):  # pragma: no cover
    """
    cuda version of spherical_harmonics_xyz
    """
    lmax = rs.lmax(Rs)
    out = real_spherical_harmonics.real_spherical_harmonics(xyz, lmax)  # real spherical harmonics are calculated for all L's up to lmax (inclusive) due to performance reasons (CUDA)
    real_spherical_harmonics.e3nn_normalization(out)  # (-1)^L, which is the same as (pi-theta) -> (-1)^(L+m) combined with 'quantum' norm (-1)^m

    if not rs.are_equal(Rs, list(range(lmax + 1))):
        out = torch.cat([out[l*l:(l+1)*(l+1)] for l in rs.extract_l(Rs)])
    return out.t()


def spherical_harmonics_alpha_beta(Rs, alpha, beta):
    """
    spherical harmonics

    :param Rs: list of L's
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    Rs = rs.simplify(Rs)
    if alpha.device.type == 'cuda' and beta.device.type == 'cuda' and not alpha.requires_grad and not beta.requires_grad and rs.lmax(Rs) <= 10 and not rsh_no_cuda:  # pragma: no cover
        xyz = torch.stack([beta.sin() * alpha.cos(), beta.sin() * alpha.sin(), beta.cos()], dim=-1)
        rsh = spherical_harmonics_xyz_cuda(Rs, xyz)
    else:
        rsh = spherical_harmonics_alpha_z_y(Rs, alpha, beta.cos(), beta.sin().abs())
    return rsh


def spherical_harmonics_xyz(Rs, xyz, expect_normalized=False):
    """
    spherical harmonics

    :param Rs: list of L's
    :param xyz: tensor of shape [..., 3]
    :param eps: epsilon for denominator of atan2
    :return: tensor of shape [..., m]

    The eps parameter is only to be used when backpropogating to coordinates xyz.
    To determine a stable eps value, we recommend benchmarking against numerical
    gradients before setting this parameter. Use the smallest epsilon that prevents NaNs.
    For some cases, we have used 1e-10. Your case may require a different value.
    Use this option with care.
    """
    Rs = rs.simplify(Rs)

    *size, _ = xyz.size()
    xyz = xyz.reshape(-1, 3)

    if not expect_normalized:
        xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)

    # use cuda implementation if possible, otherwise use cpu implementation
    if xyz.device.type == 'cuda' and not xyz.requires_grad and rs.lmax(Rs) <= 10 and not rsh_no_cuda:
        rsh = spherical_harmonics_xyz_cuda(Rs, xyz)
    else:
        alpha = torch.atan2(xyz[:, 1], xyz[:, 0])
        z = xyz[:, 2]
        y = (xyz[:, 0].pow(2) + xyz[:, 1].pow(2)).sqrt()
        rsh = spherical_harmonics_alpha_z_y(Rs, alpha, z, y)
    return rsh.reshape(*size, rs.irrep_dim(Rs))


