# pylint: disable=not-callable, no-member, invalid-name, line-too-long
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""
import math

import torch

from se3cnn.util.cache_file import cached_dirpklgz
from se3cnn.util.default_dtype import torch_default_dtype


def rot_z(gamma):
    """
    Rotation around Z axis
    """
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return gamma.new_tensor([
        [gamma.cos(), -gamma.sin(), 0],
        [gamma.sin(), gamma.cos(), 0],
        [0, 0, 1]
    ])


def rot_y(beta):
    """
    Rotation around Y axis
    """
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return beta.new_tensor([
        [beta.cos(), 0, beta.sin()],
        [0, 1, 0],
        [-beta.sin(), 0, beta.cos()]
    ])


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

    x = x / torch.norm(x, 2, -1, keepdim=True)
    beta = torch.acos(x[..., 2])
    alpha = torch.atan2(x[..., 1], x[..., 0])
    return alpha, beta


def rot_to_abc(R):
    """
    Convert rotation matrix into (alpha, beta, gamma)
    """
    x = R @ R.new_tensor([0, 0, 1])
    a, b = xyz_to_angles(x)
    R = rot(a, b, 0).t() @ R
    c = torch.atan2(R[1, 0], R[0, 0])
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
        return direct_sum(*[irr_repr(l, *abc) for mul, l, _ in normalizeRs(Rs) for _ in range(mul)])
    else:
        assert all(parity != 0 for _, _, parity in normalizeRs(Rs))
        return direct_sum(*[(p ** parity) * irr_repr(l, *abc) for mul, l, p in normalizeRs(Rs) for _ in range(mul)])


################################################################################
# Rs lists
################################################################################

def haslinearpathRs(Rs_in, l_out, p_out):
    """
    :param Rs_in: list of triplet (multiplicity, representation order, parity)
    :return: if there is a linear operation between them
    """
    for mul_in, l_in, p_in in Rs_in:
        if mul_in == 0:
            continue

        for l in range(abs(l_in - l_out), l_in + l_out + 1):
            if p_out == 0 or p_in * (-1) ** l == p_out:
                return True
    return False


def normalizeRs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
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



################################################################################
# Spherical harmonics
################################################################################

def spherical_harmonics(order, alpha, beta, dtype=None, device=None):
    """
    spherical harmonics

    :param order: int or list
    :param alpha: float or tensor of shape [...]
    :param beta: float or tensor of shape [...]
    :return: tensor of shape [m, ...]

    - compatible with irr_repr and compose
    """
    from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default
    import numpy as np

    if not isinstance(order, list):
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
    return torch.tensor(Y, dtype=dtype, device=device)


def spherical_harmonics_xyz(order, xyz, dtype=None, device=None):
    """
    spherical harmonics

    :param order: int or list
    :param xyz: tensor of shape [..., 3]
    :return: tensor of shape [m, ...]
    """
    if not isinstance(order, list):
        order = [order]

    with torch_default_dtype(torch.float64):
        alpha, beta = xyz_to_angles(xyz)  # two tensors of shape [...]
        out = spherical_harmonics(order, alpha, beta, dtype=dtype, device=device)  # [m, ...]

        # fix values when xyz = 0
        val = torch.cat([xyz.new_tensor([1 / math.sqrt(4 * math.pi)]) if l == 0 else xyz.new_zeros(2 * l + 1) for l in order])  # [m]
        out[:, xyz.norm(2, -1) == 0] = val.view(-1, 1)
        return out


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


def spherical_harmonics_onehot(lmax, alpha, beta):
    """
    approximation of a signal that is 0 everywhere except in (alpha, beta) it is one
    the higher is lmax the better is the approximation
    """
    a = sum(2 * l + 1 for l in range(lmax + 1)) / (4 * math.pi)
    return torch.cat([spherical_harmonics(l, alpha, beta) for l in range(lmax + 1)]) / a


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

def get_matrix_kernel(A, eps=1e-10):
    """
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    """
    _u, s, v = torch.svd(A)

    # A = u @ torch.diag(s) @ v.t()
    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    """
    Computes the commun kernel of all the As matrices
    """
    return get_matrix_kernel(torch.cat(As, dim=0), eps)


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

def clebsch_gordan(l1, l2, l3):
    """
    Computes the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    if torch.is_tensor(l1):
        l1 = l1.item()
    if torch.is_tensor(l2):
        l2 = l2.item()
    if torch.is_tensor(l3):
        l3 = l3.item()

    if l1 <= l2 <= l3:
        return _clebsch_gordan(l1, l2, l3)
    if l1 <= l3 <= l2:
        return _clebsch_gordan(l1, l3, l2).transpose(1, 2).contiguous()
    if l2 <= l1 <= l3:
        return _clebsch_gordan(l2, l1, l3).transpose(0, 1).contiguous()
    if l3 <= l2 <= l1:
        return _clebsch_gordan(l3, l2, l1).transpose(0, 2).contiguous()
    if l2 <= l3 <= l1:
        return _clebsch_gordan(l2, l3, l1).transpose(0, 2).transpose(1, 2).contiguous()
    if l3 <= l1 <= l2:
        return _clebsch_gordan(l3, l1, l2).transpose(0, 2).transpose(0, 1).contiguous()


@cached_dirpklgz("cache/clebsch_gordan")
def _clebsch_gordan(l1, l2, l3, _version=2):
    """
    Computes the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    # these three propositions are equivalent
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert abs(l3 - l1) <= l2 <= l3 + l1
    assert abs(l1 - l2) <= l3 <= l1 + l2

    with torch_default_dtype(torch.float64):
        n = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)

        def _DxDxD(a, b, c):
            D1 = irr_repr(l1, a, b, c)
            D2 = irr_repr(l2, a, b, c)
            D3 = irr_repr(l3, a, b, c)
            return torch.einsum('il,jm,kn->ijklmn', (D1, D2, D3)).view(n, n)

        random_angles = [
            [4.41301023, 5.56684102, 4.59384642],
            [4.93325116, 6.12697327, 4.14574096],
            [0.53878964, 4.09050444, 5.36539036],
            [2.16017393, 3.48835314, 5.55174441],
            [2.52385107, 0.29089583, 3.90040975],
        ]
        null_space = get_matrices_kernel([_DxDxD(*abc) - torch.eye(n) for abc in random_angles])

        assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
        Q = null_space[0]
        Q = Q.view(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

        if Q.sum() < 0:
            Q.neg_()

        abc = torch.rand(3)
        _Q = torch.einsum("il,jm,kn,lmn", (irr_repr(l1, *abc), irr_repr(l2, *abc), irr_repr(l3, *abc), Q))
        assert torch.allclose(Q, _Q)

    assert Q.dtype == torch.float64
    return Q  # [m1, m2, m3]



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
            [-3**.5/3, 0, 0, 0, -3**.5/3, 0, 0, 0, 12**.5/3],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0]
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(2, a, b, c) @ to5, to5 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

    return to1.type(torch.get_default_dtype()), to3.type(torch.get_default_dtype()), to5.type(torch.get_default_dtype())


def reduce_tensor_product(Rs_i, Rs_j):
    """
    Compute the orthonormal change of basis Q
    from Rs_reduced to Rs_i tensor product with Rs_j
    where Rs_reduced is a direct sum of irreducible representations

    :return: Rs_reduced, Q
    """
    with torch_default_dtype(torch.float64):
        Rs_i = normalizeRs(Rs_i)
        Rs_j = normalizeRs(Rs_j)

        n_i = sum(mul * (2 * l + 1) for mul, l, p in Rs_i)
        n_j = sum(mul * (2 * l + 1) for mul, l, p in Rs_j)
        out = torch.zeros(n_i, n_j, n_i * n_j, dtype=torch.float64)

        Rs_reduced = []
        beg = 0

        beg_i = 0
        for mul_i, l_i, p_i in Rs_i:
            n_i = mul_i * (2 * l_i + 1)

            beg_j = 0
            for mul_j, l_j, p_j in Rs_j:
                n_j = mul_j * (2 * l_j + 1)

                for l in range(abs(l_i - l_j), l_i + l_j + 1):
                    Rs_reduced.append((mul_i * mul_j, l, p_i * p_j))
                    n = mul_i * mul_j * (2 * l + 1)

                    # put sqrt(2l+1) to get an orthonormal output
                    Q = math.sqrt(2 * l + 1) * clebsch_gordan(l_i, l_j, l)  # [m_i, m_j, m]
                    I = torch.eye(mul_i * mul_j).view(mul_i, mul_j, mul_i * mul_j)  # [mul_i, mul_j, mul_i * mul_j]

                    Q = torch.einsum("ijk,mno->imjnko", (I, Q))

                    view = out[beg_i: beg_i + n_i, beg_j: beg_j + n_j, beg: beg + n]
                    view.add_(Q.view_as(view))

                    beg += n
                beg_j += n_j
            beg_i += n_i
        return Rs_reduced, out
