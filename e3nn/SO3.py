# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""
import math
from functools import lru_cache, partial

import torch
from appdirs import user_cache_dir

from e3nn.util.cache_file import cached_dirpklgz
from e3nn.util.default_dtype import torch_default_dtype

if torch.cuda.is_available():
    try:
        from e3nn import real_spherical_harmonics  # pylint: disable=no-name-in-module
    except ImportError:
        real_spherical_harmonics = None


def rot_z(gamma):
    """
    Rotation around Z axis
    """
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())

    return torch.stack([
        torch.stack([gamma.cos(),
                     -gamma.sin(),
                     gamma.new_zeros(gamma.shape)], dim=-1),
        torch.stack([gamma.sin(),
                     gamma.cos(),
                     gamma.new_zeros(gamma.shape)], dim=-1),
        torch.stack([gamma.new_zeros(gamma.shape),
                     gamma.new_zeros(gamma.shape),
                     gamma.new_ones(gamma.shape)], dim=-1)
    ], dim=-2)


def rot_y(beta):
    """
    Rotation around Y axis
    """
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return torch.stack([
        torch.stack([beta.cos(),
                     beta.new_zeros(beta.shape),
                     beta.sin()], dim=-1),
        torch.stack([beta.new_zeros(beta.shape),
                     beta.new_ones(beta.shape),
                     beta.new_zeros(beta.shape)], dim=-1),
        torch.stack([-beta.sin(),
                     beta.new_zeros(beta.shape),
                     beta.cos()], dim=-1),
    ], dim=-2)


# The following two functions (rot and xyz_to_angles) satisfies that
# rot(*xyz_to_angles([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]

def rot(alpha, beta, gamma):
    """
    ZYZ Euler angles rotation
    """
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def rand_rot():
    """
    random rotation matrix
    """
    alpha, gamma = 2 * math.pi * torch.rand(2)
    beta = torch.rand(()).mul(2).sub(1).acos()
    return rot(alpha, beta, gamma)


def angles_to_xyz(alpha, beta):
    """
    Convert (alpha, beta) into point (x, y, z) on the sphere
    """
    x = torch.sin(beta) * torch.cos(alpha)
    y = torch.sin(beta) * torch.sin(alpha)
    z = torch.cos(beta)
    return x, y, z


def xyz_to_angles(x, y=None, z=None):
    """
    Convert point (x, y, z) on the sphere into (alpha, beta)
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.get_default_dtype())

    if y is not None and z is not None:
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.get_default_dtype())
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.get_default_dtype())
        x = torch.stack([x, y, z], dim=-1)

    x = torch.nn.functional.normalize(x, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    x.masked_fill_(x < -1., -1.)                       # mitigate numerical inaccuracies from normalization
    x.masked_fill_(x > 1., 1.)

    beta = torch.acos(x[..., 2])
    alpha = torch.atan2(x[..., 1], x[..., 0])
    return alpha, beta


def rot_to_abc(R):
    """
    Convert rotation matrix into (alpha, beta, gamma)
    """
    x = R @ R.new_tensor([0, 0, 1])
    a, b = xyz_to_angles(x)
    R = rot(a, b, a.new_zeros(a.shape)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return a, b, c


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = xyz_to_angles(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def irr_repr(order, alpha, beta, gamma, dtype=None, device=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    abc = [alpha, beta, gamma]
    for i, x in enumerate(abc):
        if torch.is_tensor(x):
            abc[i] = x.item()
            if dtype is None:
                dtype = x.dtype
            if device is None:
                device = x.device
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(wigner_D_matrix(order, *abc), dtype=dtype, device=device)


def derivative_irr_repr(order, alpha, beta, gamma, dtype=None, device=None):
    """
    derivative of irreducible representation of SO3.
    returns (dDda, dDdb, dDdc)
    """
    import lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense as ph
    abc = [alpha, beta, gamma]
    for i, x in enumerate(abc):
        if torch.is_tensor(x):
            abc[i] = x.item()
            if dtype is None:
                dtype = x.dtype
            if device is None:
                device = x.device
    if dtype is None:
        dtype = torch.get_default_dtype()
    dDdabc = ph.derivative_rot_mat(*abc, l=order, J=ph.Jd[order])
    dDda, dDdb, dDdc = [torch.tensor(i, dtype=dtype, device=device) for i in dDdabc]
    return dDda, dDdb, dDdc


def rep(Rs, alpha, beta, gamma, parity=None):
    """
    Representation of O(3). Parity applied (-1)**parity times.
    """
    abc = [alpha, beta, gamma]
    if parity is None:
        return direct_sum(*[irr_repr(l, *abc) for mul, l, _ in simplifyRs(Rs) for _ in range(mul)])
    else:
        assert all(parity != 0 for _, _, parity in simplifyRs(Rs))
        return direct_sum(*[(p ** parity) * irr_repr(l, *abc) for mul, l, p in simplifyRs(Rs) for _ in range(mul)])


def selection_rule(l1, l2, lmax=None):
    """
    selection rule
    :return: list from |l1-l2|... to l1+l2
    """
    if lmax is None:
        l_max = l1 + l2
    else:
        l_max = min(lmax, l1 + l2)
    return list(range(abs(l1 - l2), l_max + 1))


################################################################################
# Rs lists
################################################################################

def haslinearpathRs(Rs_in, l_out, p_out, get_l_output=selection_rule):
    """
    :param Rs_in: list of triplet (multiplicity, representation order, parity)
    :return: if there is a linear operation between them
    """
    for mul_in, l_in, p_in in Rs_in:
        if mul_in == 0:
            continue

        for l in get_l_output(l_in, l_out):
            if p_out in (0, p_in * (-1) ** l):
                return True
    return False


def splitRs(Rs, cmul=-1):
    """
    :param Rs: [(mul, 0), (mul, 1), (mul, 2)]
    :return:   mul * [(1, 0), (1, 1), (1, 2)]
    """
    Rs = simplifyRs(Rs)
    muls = {mul for mul, _, _ in Rs}
    if cmul == -1:
        from fractions import gcd
        from functools import reduce
        cmul = reduce(gcd, muls)
    assert all(mul % cmul == 0 for mul, _, _ in Rs)

    return cmul * [(mul // cmul, l, p) for mul, l, p in Rs]


def rearrangeRs(Rs_in, Rs_out):
    """
    :return: permutation_matrix
    output = einsum('ij,j->i', permutation_matrix, input)
    """
    Rs_in, a = sortRs(Rs_in)
    Rs_out, b = sortRs(Rs_out)
    assert simplifyRs(Rs_in) == simplifyRs(Rs_out)
    return b.T @ a


def sortRs(Rs):
    """
    :return: (Rs_out, permutation_matrix)
    stable sorting of the representation by (l, p)
    output = einsum('ij,j->i', permutation_matrix, input)
    """
    Rs_in = simplifyRs(Rs)
    xs = []

    j = 0  # input offset
    for mul, l, p in Rs_in:
        d = mul * (2 * l + 1)
        xs.append((l, p, mul, j, d))
        j += d

    permutation_matrix = torch.zeros(j, j)

    Rs_out = []
    i = 0  # output offset
    for l, p, mul, j, d in sorted(xs):
        Rs_out.append((mul, l, p))
        permutation_matrix[i:i + d, j:j + d] = torch.eye(d)
        i += d

    return Rs_out, permutation_matrix


def irrep_dimRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of irreps of the representation without multiplicities
    """
    Rs = conventionRs(Rs)
    return sum(2 * l + 1 for _, l, _ in Rs)


def mul_dimRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of multiplicities of the representation
    """
    Rs = conventionRs(Rs)
    return sum(mul for mul, _, _ in Rs)


def dimRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: dimention of the representation
    """
    Rs = conventionRs(Rs)
    return sum(mul * (2 * l + 1) for mul, l, _ in Rs)


def conventionRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: conventional version of the same list which always includes parity
    """
    out = []
    for r in Rs:
        if len(r) == 2:
            mul, l = r
            p = 0
        if len(r) == 3:
            mul, l, p = r
            if p > 0:
                p = 1
            if p < 0:
                p = -1

        if mul == 0:
            continue

        out.append((mul, l, p))
    return out


def simplifyRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    out = []
    Rs = conventionRs(Rs)
    for r in Rs:
        mul, l, p = r
        if out and out[-1][1:] == (l, p):
            out[-1] = (out[-1][0] + mul, l, p)
        else:
            out.append((mul, l, p))
    return out


def formatRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    d = {
        0: "",
        1: "+",
        -1: "-",
    }
    return ",".join("{}{}{}".format("{}x".format(mul) if mul > 1 else "", l, d[p]) for mul, l, p in Rs)


def map_irrep_to_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: mapping matrix from irreps to full representation order (Rs) such
    that einsum('ij,j->i', mapping_matrix, irrep_rep) will have channel
    dimension of dimRs(Rs).

    examples: 
    Rs = [(1, 0), (2, 1), (1, 0)] will return a matrix with 0s and 1s with
    shape [dimRs(Rs), 1 + 3 + 1] such there is only one 1 in each row, but
    there can be multiple 1s per column.

    Rs = [(2, 0), (2, 1)] will return a matrix with 0s and 1s with
    shape [dimRs(Rs), 1 + 3] such there is only one 1 in each row (rep), but
    there can be multiple 1s per column (irrep).
    """
    Rs = conventionRs(Rs)
    mapping_matrix = torch.zeros(dimRs(Rs), irrep_dimRs(Rs))
    start_irrep = 0
    start_rep = 0
    for mult, L, p in Rs:
        for i in range(mult):
            irrep_slice = slice(start_irrep, start_irrep + 2 * L + 1)
            rep_slice = slice(start_rep, start_rep + 2 * L + 1)
            mapping_matrix[rep_slice, irrep_slice] = torch.eye(2 * L + 1)
            start_rep += 2 * L + 1
        start_irrep += 2 * L + 1
    return mapping_matrix  # [dimRs(Rs), irrep_dimRs(Rs)]


def map_mul_to_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: mapping matrix for multiplicity and full representation order (Rs)
    such that einsum('ij,j->i', mapping_matrix, mul_rep) will have channel
    dimension of dimRs(Rs)
    
    examples:
    Rs = [(1, 0), (2, 1), (1, 0)] will return a matrix with 0s and 1s with
    shape [dimRs(Rs), 1 + 2 + 1] such there is only one 1 in each row, but
    there can be multiple 1s per column.

    Rs = [(2, 0), (2, 1)] will return a matrix with 0s and 1s with
    shape [dimRs(Rs), 2 + 2] such there is only one 1 in each row (rep), but
    there can be multiple 1s per column (irrep).
    """
    Rs = conventionRs(Rs)
    mapping_matrix = torch.zeros(dimRs(Rs), mul_dimRs(Rs))
    start_mul = 0
    start_rep = 0
    for mult, L, p in Rs:
        for i in range(mult):
            rep_slice = slice(start_rep, start_rep + 2 * L + 1)
            mapping_matrix[rep_slice, start_mul] = 1.0
            start_mul += 1
            start_rep += 2 * L + 1
    return mapping_matrix  # [dimRs(Rs), mul_dimRs(Rs)]


def sorted_truncated_tensor_productRs(Rs_1, Rs_2, lmax):
    """
    :param Rs_1: input representation
    :param Rs_2: input representation
    :param lmax: maximum l for the output
    :return: Rs_out, change of basis

    example: sorted_truncated_tensor_productRs([(1, 1), (2, 2)], [(2, 2)], 1) = ([(4, 0, 0), (6, 1, 0)], matrix)
    """
    Rs, tensor_product = tensor_productRs(Rs_1, Rs_2, partial(selection_rule, lmax=lmax))
    Rs, sort = sortRs(Rs)
    Rs = simplifyRs(Rs)
    matrix = torch.einsum('ij,jkl->ikl', sort, tensor_product)
    return Rs, matrix


def tensor_productRs(Rs_1, Rs_2, get_l_output=selection_rule, paths=False):
    """
    Compute the orthonormal change of basis Q
    from Rs_out to Rs_1 tensor product with Rs_2
    where Rs_out is a direct sum of irreducible representations

    :return: Rs_out, Q

    Q_kij A_i B_j
    """
    Rs_1 = simplifyRs(Rs_1)
    Rs_2 = simplifyRs(Rs_2)

    Rs_out = []

    if paths:
        path_list = []
    for mul_1, l_1, p_1 in Rs_1:
        for mul_2, l_2, p_2 in Rs_2:
            for l in get_l_output(l_1, l_2):
                if paths:
                    path_list.extend([[l_1, l_2, l]] * (mul_1 * mul_2))
                Rs_out.append((mul_1 * mul_2, l, p_1 * p_2))

    Rs_out = simplifyRs(Rs_out)

    mixing_matrix = torch.zeros(dimRs(Rs_out), dimRs(Rs_1), dimRs(Rs_2))

    index_out = 0

    index_1 = 0
    for mul_1, l_1, _p_1 in Rs_1:
        dim_1 = mul_1 * (2 * l_1 + 1)

        index_2 = 0
        for mul_2, l_2, _p_2 in Rs_2:
            dim_2 = mul_2 * (2 * l_2 + 1)
            for l in get_l_output(l_1, l_2):
                dim_out = mul_1 * mul_2 * (2 * l + 1)
                C = clebsch_gordan(l, l_1, l_2, cached=True) * (2 * l + 1) ** 0.5
                I = torch.eye(mul_1 * mul_2).view(mul_1 * mul_2, mul_1, mul_2)
                m = torch.einsum("wuv,kij->wkuivj", I, C).view(dim_out, dim_1, dim_2)
                mixing_matrix[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m
                index_out += dim_out

            index_2 += dim_2
        index_1 += dim_1

    if paths:
        return Rs_out, mixing_matrix, path_list
    return Rs_out, mixing_matrix


def elementwise_tensor_productRs(Rs_1, Rs_2, get_l_output=selection_rule):
    """
    :return: Rs_out, matrix

    m_kij A_i B_j
    """
    Rs_1 = simplifyRs(Rs_1)
    Rs_2 = simplifyRs(Rs_2)

    assert sum(mul for mul, _, _ in Rs_1) == sum(mul for mul, _, _ in Rs_2)

    i = 0
    while i < len(Rs_1):
        mul_1, l_1, p_1 = Rs_1[i]
        mul_2, l_2, p_2 = Rs_2[i]

        if mul_1 < mul_2:
            Rs_2[i] = (mul_1, l_2, p_2)
            Rs_2.insert(i + 1, (mul_2 - mul_1, l_2, p_2))

        if mul_2 < mul_1:
            Rs_1[i] = (mul_2, l_1, p_1)
            Rs_1.insert(i + 1, (mul_1 - mul_2, l_1, p_1))
        i += 1

    Rs_out = []
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
        assert mul == mul_2
        for l in get_l_output(l_1, l_2):
            Rs_out.append((mul, l, p_1 * p_2))

    Rs_out = simplifyRs(Rs_out)

    mixing_matrix = torch.zeros(dimRs(Rs_out), dimRs(Rs_1), dimRs(Rs_2))

    index_out = 0
    index_1 = 0
    index_2 = 0
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
        assert mul == mul_2
        dim_1 = mul * (2 * l_1 + 1)
        dim_2 = mul * (2 * l_2 + 1)

        for l in get_l_output(l_1, l_2):
            dim_out = mul * (2 * l + 1)
            C = clebsch_gordan(l, l_1, l_2, cached=True) * (2 * l + 1) ** 0.5
            I = torch.einsum("uv,wu->wuv", torch.eye(mul), torch.eye(mul))
            m = torch.einsum("wuv,kij->wkuivj", I, C).view(dim_out, dim_1, dim_2)
            mixing_matrix[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m
            index_out += dim_out

        index_1 += dim_1
        index_2 += dim_2

    return Rs_out, mixing_matrix

################################################################################
# Spherical harmonics
################################################################################


def spherical_harmonics(order, alpha, beta, sph_last=False, dtype=None, device=None):
    """
    spherical harmonics

    :param order: int or list
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :param sph_last: return the spherical harmonics in the last channel
    :param dtype:
    :param device:
    :return: tensor of shape [m, ...] (or [..., m] if sph_last)

    - compatible with irr_repr and compose
    """
    from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default
    import numpy as np

    try:
        order = list(order)
    except TypeError:
        order = [order]

    if dtype is None and torch.is_tensor(alpha):
        dtype = alpha.dtype
    if dtype is None and torch.is_tensor(beta):
        dtype = beta.dtype
    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None and torch.is_tensor(alpha):
        device = alpha.device
    if device is None and torch.is_tensor(beta):
        device = beta.device

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, dtype=torch.float64)
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.float64)

    Js = np.concatenate([J * np.ones(2 * J + 1) for J in order], 0)
    Ms = np.concatenate([np.arange(-J, J + 1, 1) for J in order], 0)
    Js = Js.reshape(-1, *[1] * alpha.dim())
    Ms = Ms.reshape(-1, *[1] * alpha.dim())
    alpha = alpha.unsqueeze(0)
    beta = beta.unsqueeze(0)
    Y = sh(Js, Ms, math.pi - beta.cpu().numpy(), alpha.cpu().numpy())
    if sph_last:
        rank = len(Y.shape)
        return torch.tensor(Y, dtype=dtype, device=device).permute(*range(1, rank), 0).contiguous()
    else:
        return torch.tensor(Y, dtype=dtype, device=device)


def spherical_harmonics_xyz(order, xyz, sph_last=False, dtype=None, device=None):
    """
    spherical harmonics

    :param order: int or list
    :param xyz: tensor of shape [..., 3]
    :param sph_last: return the spherical harmonics in the last channel
    :param dtype:
    :param device:
    :return: tensor of shape [m, ...] (or [..., m] if sph_last)
    """
    try:
        order = list(order)
    except TypeError:
        order = [order]

    if dtype is None and torch.is_tensor(xyz):
        dtype = xyz.dtype
    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None and torch.is_tensor(xyz):
        device = xyz.device

    if not torch.is_tensor(xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64)

    with torch_default_dtype(torch.float64):
        if device.type == 'cuda' and max(order) <= 10 and real_spherical_harmonics is not None:
            *size, _ = xyz.size()
            xyz = xyz.view(-1, 3)
            max_l = max(order)
            out = xyz.new_empty(((max_l + 1) * (max_l + 1), xyz.size(0)))  # [ filters, batch_size]
            xyz_unit = torch.nn.functional.normalize(xyz, p=2, dim=-1)
            real_spherical_harmonics.rsh(out, xyz_unit)
            # (-1)^L same as (pi-theta) -> (-1)^(L+m) and 'quantum' norm (-1)^m combined  # h - halved
            norm_coef = [elem for lh in range((max_l + 1) // 2) for elem in [1.] * (4 * lh + 1) + [-1.] * (4 * lh + 3)]
            if max_l % 2 == 0:
                norm_coef.extend([1.] * (2 * max_l + 1))
            norm_coef = torch.tensor(norm_coef, device=device).unsqueeze(1)
            out.mul_(norm_coef)
            if order != list(range(max_l + 1)):
                keep_rows = torch.zeros(out.size(0), dtype=torch.bool)
                for l in order:
                    keep_rows[(l * l):((l + 1) * (l + 1))].fill_(True)
                out = out[keep_rows.to(device)]
            out = out.view(-1, *size)
        else:
            alpha, beta = xyz_to_angles(xyz)  # two tensors of shape [...]
            out = spherical_harmonics(order, alpha, beta)  # [m, ...]

            # fix values when xyz = 0
            val = xyz.new_tensor([1 / math.sqrt(4 * math.pi)])
            val = torch.cat([val if l == 0 else xyz.new_zeros(2 * l + 1) for l in order])  # [m]
            out[:, xyz.norm(2, -1) == 0] = val.view(-1, 1)

        if sph_last:
            rank = len(out.shape)
            return out.to(dtype=dtype, device=device).permute(*range(1, rank), 0).contiguous()
        else:
            return out.to(dtype=dtype, device=device)


def _legendre(order, z):
    """
    associated Legendre polynomials

    :param order: int
    :param z: tensor of shape [A]
    :return: tensor of shape [m, A]
    """
    fac = math.factorial(order)
    sqz2 = (1 - z ** 2) ** 0.5
    hsqz2 = 0.5 * sqz2
    ihsqz2 = z / hsqz2

    if order == 0:
        return z.new_ones(1, *z.size())
    if order == 1:
        return torch.stack([-0.5 * sqz2, z, sqz2])

    plm = [(1 - 2 * abs(order - 2 * order // 2)) * hsqz2 ** order / fac]
    plm.append(-plm[0] * order * ihsqz2)
    for mr in range(1, 2 * order):
        plm.append((mr - order) * ihsqz2 * plm[mr] - (2 * order - mr + 1) * mr * plm[mr - 1])
    return torch.stack(plm)


def legendre(order, z):
    """
    associated Legendre polynomials

    :param order: int
    :param z: tensor of shape [A]
    :return: tensor of shape [l * m, A]
    """
    if not isinstance(order, list):
        order = [order]
    return torch.cat([_legendre(J, z) for J in order], dim=0)  # [l * m, A]


def _spherical_harmonics_xyz_backwardable(order, xyz, eps):
    """
    spherical harmonics

    :param order: int
    :param xyz: tensor of shape [A, 3]
    :return: tensor of shape [m, A]
    """
    norm = torch.norm(xyz, 2, -1, keepdim=True)
    # Using this eps and masking out spherical harmonics from radii < eps
    # are both crucial to stability.
    xyz = xyz / (norm + eps)

    plm = legendre(order, xyz[..., 2])  # [m, A]

    m = torch.arange(-order, order + 1, dtype=xyz.dtype, device=xyz.device)
    m = m.view(-1, *(1, ) * (xyz.dim() - 1))  # [m, 1...]
    sm = 1 - m % 2 * 2  # [m, 1...]

    phi = torch.atan2(xyz[..., 1], xyz[..., 0]).unsqueeze(0)  # [1, A]
    exr = torch.cos(m * phi)  # [m, A]
    exi = torch.sin(-m * phi)  # [-m, A]

    if order == 0:
        prefactor = 1.
    else:
        prefactor = torch.cat([
            2 ** 0.5 * sm[:order] * exi[:order],
            xyz.new_ones(1, *xyz.size()[:-1]),
            2 ** 0.5 * exr[-order:],
        ])

    if order == 1:
        prefactor *= -1

    quantum = [((2 * order + 1) / (4 * math.pi) * math.factorial(order - m) / math.factorial(order + m)) ** 0.5 for m in m]
    quantum = xyz.new_tensor(quantum).view(-1, *(1, ) * (xyz.dim() - 1))  # [m, 1...]

    out = prefactor * quantum * plm  # [m, A]

    # fix values when xyz = 0
    out[:, norm.squeeze(-1) < eps] = 1 / math.sqrt(4 * math.pi) if order == 0 else 0.

    return out


def spherical_harmonics_xyz_backwardable(order, xyz, eps=1e-8):
    """
    spherical harmonics

    :param order: int
    :param xyz: tensor of shape [A, 3]
    :return: tensor of shape [l * m, A]
    """
    if not isinstance(order, list):
        order = [order]
    return torch.cat([_spherical_harmonics_xyz_backwardable(J, xyz, eps) for J in order], dim=0)  # [m, A]


def spherical_harmonics_dirac(lmax, alpha, beta, sph_last=False, dtype=None, device=None):
    """
    approximation of a signal that is 0 everywhere except on the angle (alpha, beta) where it is one.
    the higher is lmax the better is the approximation
    """
    a = sum(2 * l + 1 for l in range(lmax + 1)) / (4 * math.pi)

    if sph_last:
        onehot = torch.cat([spherical_harmonics(l, alpha, beta, dtype=dtype, device=device) for l in range(lmax + 1)]) / a
        rank = len(onehot.shape)
        return onehot.permute(*range(1, rank), 0).contiguous()
    else:
        return torch.cat([spherical_harmonics(l, alpha, beta, dtype=dtype, device=device) for l in range(lmax + 1)]) / a


def spherical_harmonics_coeff_to_sphere(coeff, alpha, beta):
    """
    Evaluate the signal on the sphere
    """
    from itertools import count
    s = 0
    i = 0
    for l in count():
        d = 2 * l + 1
        if len(coeff) < i + d:
            break
        c = coeff[i: i + d]
        i += d

        s += torch.einsum("i,i...->...", (c, spherical_harmonics(l, alpha, beta)))
    return s


################################################################################
# Linear algebra
################################################################################

def kron(x, y):
    """
    Kroneker product between two matrices
    """
    assert x.dim() == 2
    assert y.dim() == 2
    return torch.einsum("ij,kl->ikjl", (x, y)).contiguous().view(x.size(0) * y.size(0), x.size(1) * y.size(1))


def direct_sum(*matrices):
    """
    Direct sum of matrices, put them in the diagonal
    """
    m = sum(x.size(0) for x in matrices)
    n = sum(x.size(1) for x in matrices)
    out = matrices[0].new_zeros(m, n)
    i, j = 0, 0
    for x in matrices:
        m, n = x.size()
        out[i: i + m, j: j + n] = x
        i += m
        j += n
    return out


################################################################################
# Clebsch Gordan
################################################################################

def clebsch_gordan(l1, l2, l3, cached=False, dtype=None, device=None, like=None):
    """
    Computes the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    if dtype is None:
        if like is not None:
            dtype = like.dtype
        else:
            dtype = torch.get_default_dtype()
    if device is None:
        if like is not None:
            device = like.device
        else:
            device = 'cpu'

    if cached:
        return _cached_clebsch_gordan(l1, l2, l3, dtype, device)
    return _clebsch_gordan(l1, l2, l3).to(dtype=dtype, device=device)


@lru_cache(maxsize=None)
def _cached_clebsch_gordan(l1, l2, l3, dtype, device):
    return _clebsch_gordan(l1, l2, l3).to(dtype=dtype, device=device)


def _clebsch_gordan(l1, l2, l3):
    if torch.is_tensor(l1):
        l1 = l1.item()
    if torch.is_tensor(l2):
        l2 = l2.item()
    if torch.is_tensor(l3):
        l3 = l3.item()

    if l1 <= l2 <= l3:
        return __clebsch_gordan(l1, l2, l3)
    if l1 <= l3 <= l2:
        return __clebsch_gordan(l1, l3, l2).transpose(1, 2).contiguous()
    if l2 <= l1 <= l3:
        return __clebsch_gordan(l2, l1, l3).transpose(0, 1).contiguous()
    if l3 <= l2 <= l1:
        return __clebsch_gordan(l3, l2, l1).transpose(0, 2).contiguous()
    if l2 <= l3 <= l1:
        return __clebsch_gordan(l2, l3, l1).transpose(0, 2).transpose(1, 2).contiguous()
    if l3 <= l1 <= l2:
        return __clebsch_gordan(l3, l1, l2).transpose(0, 2).transpose(0, 1).contiguous()


@cached_dirpklgz(user_cache_dir("e3nn/clebsch_gordan"))
def __clebsch_gordan(l1, l2, l3, _version=4):
    """
    Computes the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    # these three propositions are equivalent
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert abs(l3 - l1) <= l2 <= l3 + l1
    assert abs(l1 - l2) <= l3 <= l1 + l2

    with torch_default_dtype(torch.float64):
        null_space = _get_d_null_space(l1, l2, l3)

        assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
        Q = null_space[0]
        Q = Q.view(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

        if next(x for x in Q.flatten() if x.abs() > 1e-10 * Q.abs().max()) < 0:
            Q.neg_()

        abc = torch.rand(3)
        _Q = torch.einsum("il,jm,kn,lmn", (irr_repr(l1, *abc), irr_repr(l2, *abc), irr_repr(l3, *abc), Q))
        assert torch.allclose(Q, _Q)

    assert Q.dtype == torch.float64
    return Q  # [m1, m2, m3]


def _get_d_null_space(l1, l2, l3, eps=1e-10):
    import scipy
    import scipy.linalg
    import gc

    def _DxDxD(a, b, c):
        D1 = irr_repr(l1, a, b, c)
        D2 = irr_repr(l2, a, b, c)
        D3 = irr_repr(l3, a, b, c)
        return torch.einsum('il,jm,kn->ijklmn', (D1, D2, D3)).view(n, n)

    n = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)
    random_angles = [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.29089583, 3.90040975],
    ]

    B = torch.zeros((n, n))                                                                             # preallocate memory
    for abc in random_angles:                                                                           # expand block matrix multiplication with its transpose
        D = _DxDxD(*abc) - torch.eye(n)
        B += torch.matmul(D.t(), D)                                                                     # B = sum_i { D^T_i @ D_i }
        del D
        gc.collect()

    # ask for one (smallest) eigenvalue/eigenvector pair if there is only one exists, otherwise ask for two
    s, v = scipy.linalg.eigh(B.numpy(), eigvals=(0, min(1, n - 1)), overwrite_a=True)
    del B
    gc.collect()

    kernel = v.T[s < eps]
    return torch.from_numpy(kernel)


################################################################################
# Change of basis
################################################################################

def xyz_vector_basis_to_spherical_basis(check=True):
    """
    to convert a vector [x, y, z] transforming with rot(a, b, c)
    into a vector transforming with irr_repr(1, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        A = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)
        if check:
            assert all(torch.allclose(irr_repr(1, a, b, c) @ A, A @ rot(a, b, c)) for a, b, c in torch.rand(10, 3))
    return A.type(torch.get_default_dtype())


def spherical_basis_vector_to_xyz_basis(check=True):
    """
    to convert a vector transforming with irr_repr(1, a, b, c)
    into a vector [x, y, z] transforming with rot(a, b, c)
    see assert for usage

    Inverse of xyz_vector_basis_to_spherical_basis
    """
    with torch_default_dtype(torch.float64):
        A = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
        if check:
            assert all(torch.allclose(A @ irr_repr(1, a, b, c), rot(a, b, c) @ A) for a, b, c in torch.rand(10, 3))
    return A.type(torch.get_default_dtype())


def tensor3x3_repr(a, b, c):
    """
    representation of 3x3 tensors
    T --> R T R^t
    """
    r = rot(a, b, c)
    return kron(r, r)


def tensor3x3_repr_basis_to_spherical_basis():
    """
    to convert a 3x3 tensor transforming with tensor3x3_repr(a, b, c)
    into its 1 + 3 + 5 component transforming with irr_repr(0, a, b, c), irr_repr(1, a, b, c), irr_repr(3, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        to1 = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(0, a, b, c) @ to1, to1 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to3 = torch.tensor([
            [0, 0, -1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(1, a, b, c) @ to3, to3 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to5 = torch.tensor([
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [-3**.5 / 3, 0, 0, 0, -3**.5 / 3, 0, 0, 0, 12**.5 / 3],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0]
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(2, a, b, c) @ to5, to5 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

    return to1.type(torch.get_default_dtype()), to3.type(torch.get_default_dtype()), to5.type(torch.get_default_dtype())
