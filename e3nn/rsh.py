# pylint: disable=not-callable, no-member, invalid-name, line-too-long
"""
Real Spherical Harmonics equivariant with respect to o3.rot and o3.irr_repr
"""
import math

import torch
from scipy import special


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

    :param ls: list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    return torch.cat([_legendre(l, z) for l in ls], dim=-1)  # [..., l * m]


def spherical_harmonics_beta(ls, cosbeta):
    """
    the cosbeta componant of the spherical harmonics
    (useful to perform fourier transform)

    :param cosbeta: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    size = cosbeta.shape
    cosbeta = cosbeta.reshape(-1)

    output = []
    for l in ls:
        quantum = [((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) ** 0.5 for m in range(-l, l + 1)]
        quantum = torch.tensor(quantum).to(cosbeta)  # [m]
        out = (-1) ** l * quantum * legendre([l], cosbeta)  # [batch, m]
        output += [out]
    output = torch.cat(output, dim=-1)
    return output.reshape(*size, output.shape[1])  # [..., l * m]


def spherical_harmonics_alpha(l, alpha):
    """
    the alpha componant of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    size = alpha.shape
    alpha = alpha.reshape(-1, 1)  # [batch, 1]

    if l == 0:
        out = torch.ones_like(alpha)
    else:
        m = torch.arange(1, l + 1).flip(0).to(alpha)  # [l, l-1, l-2, ..., 1]
        sin = torch.sin(((-1)**m * m) * alpha)  # [batch, m]

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

    output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], beta.cos()) for l in ls]
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
    output = [spherical_harmonics_alpha(l, alpha) * spherical_harmonics_beta([l], cosbeta) for l in ls]
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
