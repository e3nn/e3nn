# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg, too-many-lines, import-outside-toplevel
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""
import gc
import math
from functools import lru_cache

import lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense as ph
import scipy
import scipy.linalg
import torch
from appdirs import user_cache_dir
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

from e3nn.util.cache_file import cached_dirpklgz
from e3nn.util.default_dtype import torch_default_dtype


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
    return rot(*rand_angles())


def rand_angles():
    """
    random rotation angles
    """
    alpha, gamma = 2 * math.pi * torch.rand(2)
    beta = torch.rand(()).mul(2).sub(1).acos()
    return alpha, beta, gamma


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
    derivative of irreducible representation of SO3
    returns (dDda, dDdb, dDdc)
    """
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


def selection_rule(l1, _p1, l2, _p2, lmax=None, lfilter=None):
    """
    selection rule
    :return: list from |l1-l2|... to l1+l2
    """
    if lmax is None:
        l_max = l1 + l2
    else:
        l_max = min(lmax, l1 + l2)
    ls = list(range(abs(l1 - l2), l_max + 1))
    if lfilter is not None:
        ls = list(filter(lfilter, ls))
    return ls


def selection_rule_in_out_sh(l_in, p_in, l_out, p_out, lmax=None):
    """
    all possible spherical harmonics such that
    Input * SH = Output
    """
    return [l for l in selection_rule(l_in, p_in, l_out, p_out, lmax) if p_out in [0, p_in * (-1) ** l]]


################################################################################
# Linear algebra
################################################################################

def kron(x, y):
    """
    Kroneker product between two matrices
    """
    assert x.dim() == 2
    assert y.dim() == 2
    return torch.einsum("ij,kl->ikjl", (x, y)).reshape(x.size(0) * y.size(0), x.size(1) * y.size(1))


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
# 3j symbol
################################################################################

def wigner_3j(l1, l2, l3, cached=False, dtype=None, device=None, like=None):
    """
    Computes the 3-j symbol
    https://en.wikipedia.org/wiki/3-j_symbol

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    if torch.is_tensor(l1):
        l1 = l1.item()
    if torch.is_tensor(l2):
        l2 = l2.item()
    if torch.is_tensor(l3):
        l3 = l3.item()

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

    # return a clone to avoid that the user modifies the matrices in-place
    if cached:
        return _cached_wigner_3j(l1, l2, l3, dtype, device).clone()
    return _wigner_3j(l1, l2, l3).to(dtype=dtype, device=device).clone()


@lru_cache(maxsize=None)
def _cached_wigner_3j(l1, l2, l3, dtype, device):
    return _wigner_3j(l1, l2, l3).to(dtype=dtype, device=device)


def _wigner_3j(l1, l2, l3):
    if l1 <= l2 <= l3:
        return __wigner_3j(l1, l2, l3)
    if l1 <= l3 <= l2:
        return __wigner_3j(l1, l3, l2).transpose(1, 2) * (-1) ** (l1 + l2 + l3)
    if l2 <= l1 <= l3:
        return __wigner_3j(l2, l1, l3).transpose(0, 1) * (-1) ** (l1 + l2 + l3)
    if l3 <= l2 <= l1:
        return __wigner_3j(l3, l2, l1).transpose(0, 2) * (-1) ** (l1 + l2 + l3)
    if l2 <= l3 <= l1:
        return __wigner_3j(l2, l3, l1).transpose(0, 2).transpose(1, 2)
    if l3 <= l1 <= l2:
        return __wigner_3j(l3, l1, l2).transpose(0, 2).transpose(0, 1)


@cached_dirpklgz(user_cache_dir("e3nn/wigner_3j"))
def __wigner_3j(l1, l2, l3, _version=1):
    """
    Computes the 3-j symbol
    https://en.wikipedia.org/wiki/3-j_symbol

    Closely related to the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    # these three propositions are equivalent
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert abs(l3 - l1) <= l2 <= l3 + l1
    assert abs(l1 - l2) <= l3 <= l1 + l2

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

    with torch_default_dtype(torch.float64):
        B = torch.zeros((n, n))
        for abc in random_angles:
            D = _DxDxD(*abc) - torch.eye(n)
            B += D.T @ D
            del D
            gc.collect()

    # ask for one (smallest) eigenvalue/eigenvector pair if there is only one exists, otherwise ask for two
    s, v = scipy.linalg.eigh(B.numpy(), eigvals=(0, min(1, n - 1)), overwrite_a=True)
    del B
    gc.collect()

    kernel = v.T[s < 1e-10]
    null_space = torch.from_numpy(kernel)

    assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
    Q = null_space[0]
    Q = Q.view(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

    if next(x for x in Q.flatten() if x.abs() > 1e-10 * Q.abs().max()) < 0:
        Q.neg_()

    with torch_default_dtype(torch.float64):
        abc = rand_angles()
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
            [-3**.5 / 3, 0, 0, 0, -3**.5 / 3, 0, 0, 0, 12**.5 / 3],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0]
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(2, a, b, c) @ to5, to5 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

    return to1.type(torch.get_default_dtype()), to3.type(torch.get_default_dtype()), to5.type(torch.get_default_dtype())
