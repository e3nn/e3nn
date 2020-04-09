# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg, too-many-lines
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math

import torch
from scipy import special

from e3nn.util.default_dtype import torch_default_dtype


def spherical_harmonics_expand_matrix(lmax):
    """
    :return: tensor [l, m, l * m]
    """
    m = torch.zeros(lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in range(lmax + 1)))
    i = 0
    for l in range(lmax + 1):
        m[l, lmax - l: lmax + l + 1, i:i + 2 * l + 1] = torch.eye(2 * l + 1)
        i += 2 * l + 1
    return m


def _legendre(l, z):
    """
    associated Legendre polynomials

    :param l: int
    :param z: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    if not z.requires_grad:
        return torch.stack([(-1)**m * torch.tensor(special.lpmv(m, l, z.cpu().double().numpy())).to(z) for m in range(-l, l + 1)], dim=-1)

    fac = math.factorial(l)
    sqz2 = (1 - z ** 2) ** 0.5
    hsqz2 = 0.5 * sqz2
    ihsqz2 = z / hsqz2

    if l == 0:
        return z.new_ones(*z.size(), 1)
    if l == 1:
        return torch.stack([-0.5 * sqz2, z, sqz2], dim=-1)

    plm = [(1 - 2 * abs(l - 2 * l // 2)) * hsqz2 ** l / fac]
    plm.append(-plm[0] * l * ihsqz2)
    for mr in range(1, l):
        plm.append((mr - l) * ihsqz2 * plm[mr] - (2 * l - mr + 1) * mr * plm[mr - 1])
    plm = torch.stack(plm, dim=-1)  # [..., m]
    c = torch.tensor([(-1)**m * (math.factorial(l + m) / math.factorial(l - m)) for m in range(1, l + 1)])
    plm = torch.cat([plm, plm[..., :-1].flip(-1) * c], dim=-1)
    return plm * (-1) ** l


def legendre(ls, z):
    """
    associated Legendre polynomials

    :param ls: int or list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    if not isinstance(ls, list):
        ls = [ls]
    return torch.cat([_legendre(l, z) for l in ls], dim=-1)  # [..., l * m]


def spherical_harmonics_beta(ls, cosbeta):
    """
    the cosbeta componant of the spherical harmonics
    (useful to perform fourier transform)

    :param cosbeta: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    size = cosbeta.shape
    cosbeta = cosbeta.view(-1)

    output = []
    for l in ls:
        quantum = [((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) ** 0.5 for m in range(-l, l + 1)]
        quantum = torch.tensor(quantum)  # [m]
        out = (-1) ** l * quantum * legendre(l, cosbeta)  # [batch, m]
        output += [out]
    output = torch.cat(output, dim=-1)
    return output.view(*size, -1)  # [..., l * m]


def spherical_harmonics_alpha(l, alpha):
    """
    the alpha componant of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    size = alpha.shape
    alpha = alpha.view(-1, 1)  # [batch, 1]

    m = torch.arange(-l, l + 1)  # [m]
    sm = 1 - m % 2 * 2  # [m]  = (-1)**m

    exr = torch.cos(m * alpha)  # [batch, m]
    exi = torch.sin(-m * alpha)  # [batch, -m]

    if l == 0:
        out = torch.ones_like(alpha)
    else:
        out = torch.cat([
            2 ** 0.5 * sm[:l] * exi[:, :l],
            torch.ones_like(alpha),
            2 ** 0.5 * exr[:, -l:],
        ], dim=-1)

    return out.view(*size, -1)  # [..., m]


def spherical_harmonics_alpha_beta(ls, alpha, beta):
    """
    spherical harmonics

    :param ls: list
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], beta.cos()) for l in ls]
    return torch.cat(output, dim=-1)


def spherical_harmonics_xyz(ls, xyz):
    """
    spherical harmonics

    :param ls: list of int
    :param xyz: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """

    with torch_default_dtype(torch.float64):
        norm = torch.norm(xyz, 2, -1, keepdim=True)
        xyz = xyz / norm
        alpha = torch.atan2(xyz[..., 1], xyz[..., 0])  # [...]
        cosbeta = xyz[..., 2]  # [...]
        output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], cosbeta) for l in ls]
        return torch.cat(output, dim=-1)


def spherical_harmonics_dirac(lmax, alpha, beta):
    """
    approximation of a signal that is 0 everywhere except on the angle (alpha, beta) where it is one.
    the higher is lmax the better is the approximation
    """
    ls = list(range(lmax + 1))
    a = sum(2 * l + 1 for l in ls) / (4 * math.pi)
    return spherical_harmonics_alpha_beta(ls, torch.tensor(alpha), torch.tensor(beta)) / a


def spherical_harmonics_coeff_to_sphere(coeff, alpha, beta):
    """
    Evaluate the signal on the sphere
    """
    lmax = round(coeff.shape[-1] ** 0.5) - 1
    ls = list(range(lmax + 1))
    sh = spherical_harmonics_alpha_beta(ls, alpha, beta)
    return torch.einsum('...i,i->...', sh, coeff)
