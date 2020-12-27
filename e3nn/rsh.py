# pylint: disable=not-callable, no-member, invalid-name, line-too-long, arguments-differ
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math
from functools import lru_cache

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn import rs
from e3nn.util.eval_code import eval_code

from e3nn import real_spherical_harmonics


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
    assert all(p in [0, (-1)**l] for _, l, p in Rs)
    ls = [l for mul, l, _ in Rs for _ in range(mul)]
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


def spherical_harmonics_alpha_beta(Rs, alpha, beta):
    """
    spherical harmonics
    :param Rs: list of L's
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    sh = None
    if rs.lmax(Rs) <= 10:
        xyz = torch.stack([beta.sin() * alpha.cos(), beta.sin() * alpha.sin(), beta.cos()], dim=-1)
        lmax = rs.lmax(Rs)
        sh = rsh_optimized(xyz, lmax, e3nn_normalization=True)
        if not rs.are_equal(Rs, list(range(lmax + 1))):
            sh = torch.cat([sh[l * l: (l + 1) * (l + 1)] for mul, l, _ in Rs for _ in range(mul)])
    else:
        sh = spherical_harmonics_alpha_z_y(Rs, alpha, beta.cos(), beta.sin().abs())
    return sh


def spherical_harmonics_alpha_z_y(Rs, alpha, z, y):
    """
    cpu version of spherical_harmonics_alpha_beta
    """
    Rs = rs.simplify(Rs)
    sha = spherical_harmonics_alpha(rs.lmax(Rs), alpha.flatten())  # [z, m]
    shz = spherical_harmonics_z(Rs, z.flatten(), y.flatten())  # [z, l * m]
    out = mul_m_lm(Rs, sha, shz)
    return out.reshape(alpha.shape + (shz.shape[1],))


@lru_cache()
def _rep_zx(Rs, dtype, device):
    o = torch.zeros((), dtype=dtype, device=device)
    return rs.rep(Rs, o, -math.pi / 2, o)


def spherical_harmonics_xyz(Rs, xyz, normalization='none'):
    """
    spherical harmonics
    :param Rs: list of L's
    :param xyz: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """
    sh = None
    Rs = rs.simplify(Rs)
    *size, _ = xyz.shape
    xyz = xyz.reshape(-1, 3)

    if rs.lmax(Rs) <= 10:
        xyz = torch.nn.functional.normalize(xyz, p=2, dim=1)
        lmax = rs.lmax(Rs)
        sh = rsh_optimized(xyz, lmax, e3nn_normalization=True)

        # rsh_optimized returns real spherical harmonics for all rotation orders from 0 to lmax inclusive
        # gaps in requested list of rotation orders are unusual, but possible
        if not rs.are_equal(Rs, list(range(lmax + 1))):
            sh = torch.cat([sh[l*l: (l+1)*(l+1)] for mul, l, _ in Rs for _ in range(mul)])
    else:
        # normalize coordinates and filter out 0-inputs
        d = torch.norm(xyz, 2, dim=1)
        xyz = xyz[d > 0]
        xyz = xyz / d[d > 0, None]

        # if z > x, rotate x-axis with z-axis
        s = xyz[:, 2].abs() > xyz[:, 0].abs()
        xyz[s] = xyz[s] @ xyz.new_tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # calculate real spherical harmonics
        alpha = torch.atan2(xyz[:, 1], xyz[:, 0])
        z = xyz[:, 2]
        y = xyz[:, :2].norm(dim=1)
        sh = spherical_harmonics_alpha_z_y(Rs, alpha, z, y)

        # rotate back
        sh[s] = sh[s] @ _rep_zx(tuple(Rs), xyz.dtype, xyz.device)

        # handle special case of 0-inputs
        if len(d) > len(sh):
            out = sh.new_zeros(len(d), sh.shape[1])
            out[d == 0] = math.sqrt(1 / (4 * math.pi)) * torch.cat([sh.new_ones(1) if l == 0 else sh.new_zeros(2 * l + 1) for mul, l, p in Rs for _ in range(mul)])
            out[d > 0] = sh
            sh = out

    if normalization == 'component':
        sh.mul_(math.sqrt(4 * math.pi))
    elif normalization == 'norm':
        sh.mul_(torch.cat([math.sqrt(4 * math.pi / (2 * l + 1)) * sh.new_ones(2 * l + 1) for mul, l, p in Rs for _ in range(mul)]))
    return sh.reshape(*size, sh.shape[1])


class RSH(torch.autograd.Function):
    """
    xyz coordinates are expected to be already normalized.
    e3nn_normalization is (-1)^L, it comes from combination if (pi-theta) -> (-1)^(L+m) and 'quantum' norm -> (-1)^m.
    """
    @staticmethod
    def forward(ctx, xyz, lmax, e3nn_normalization):
        Y = real_spherical_harmonics.rsh(xyz, lmax)
        if e3nn_normalization:
            real_spherical_harmonics.e3nn_normalization(Y)
        if xyz.requires_grad:
            ctx.save_for_backward(xyz)
            ctx.lmax = lmax
            ctx.e3nn_normalization = e3nn_normalization
        return Y.t().contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.clone() # uncomment for gradcheck test - particularity of e3nn_normalization being in-place operation
        if ctx.e3nn_normalization:
            e3nn_normalization(grad_output)
        xyz, = ctx.saved_tensors
        derivatives = drsh(xyz, ctx.lmax)
        grad_xyz = (derivatives * grad_output.unsqueeze(2).expand(-1, -1, 3)).sum(dim=0)
        return grad_xyz, None, None


def rsh_optimized(xyz, lmax, e3nn_normalization=False):
    # apply does not support keyword arguments
    if e3nn_normalization:
        return RSH.apply(xyz, lmax, True)
    else:
        return RSH.apply(xyz, lmax, False)

