# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg, too-many-lines, import-outside-toplevel
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""
import gc
import math
import os
from functools import lru_cache
from typing import Callable, List, Tuple

import lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense as ph
import scipy
import scipy.linalg
import torch
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

from e3nn.util.cache_file import cached_picklesjar
from e3nn.util.default_dtype import torch_default_dtype


def rot_z(gamma, dtype=None, device=None):
    """
    Rotation around Z axis
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=dtype, device=device)
    else:
        gamma = gamma.to(dtype=dtype, device=device)

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


def rot_y(beta, dtype=None, device=None):
    """
    Rotation around Y axis
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(dtype=dtype, device=device)

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

def rot(alpha, beta, gamma, dtype=None, device=None):
    """
    ZYZ Euler angles rotation
    """
    return rot_z(alpha, dtype, device) @ rot_y(beta, dtype, device) @ rot_z(gamma, dtype, device)


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
    return torch.stack([x, y, z], dim=-1)


def xyz_to_angles(x, y=None, z=None):
    """
    Convert point (x, y, z) on the sphere into (alpha, beta)
    """
    if y is not None and z is not None:
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


def compose_with_parity(a1, b1, c1, p1, a2, b2, c2, p2):
    """
    (a, b, c, p) = (a1, b1, c1, p1) composed with (a2, b2, c2, p2)
    """
    return compose(a1, b1, c1, a2, b2, c2) + ((p1 + p2) % 2,)


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


TY_SELECTION_RULE = Callable[[int, int, int, int], List[int]]


def selection_rule(l1: int, _p1: int, l2: int, _p2: int, lmax=None, lfilter=None) -> List[int]:
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


def selection_rule_in_out_sh(l_in: int, p_in: int, l_out: int, p_out: int, lmax=None) -> List[int]:
    """
    all possible spherical harmonics such that
    Input * SH = Output
    """
    return [l for l in selection_rule(l_in, p_in, l_out, p_out, lmax) if p_out in [0, p_in * (-1) ** l]]


################################################################################
# Linear algebra
################################################################################

def kron(*matrices):
    """
    Kroneker product between matrices
    """
    for m in matrices:
        assert m.dim() == 2

    if len(matrices) == 0:
        return torch.ones(1, 1)
    if len(matrices) == 1:
        return matrices[0]

    x, y, *matrices = matrices
    z = torch.einsum("ij,kl->ikjl", x, y).reshape(x.size(0) * y.size(0), x.size(1) * y.size(1))

    if matrices:
        return kron(z, *matrices)
    return z


def direct_sum(*matrices):
    """
    Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(*total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
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
    assert isinstance(l1, int)
    assert isinstance(l2, int)
    assert isinstance(l3, int)

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


@cached_picklesjar(os.path.join(os.path.dirname(__file__), 'cache/wigner_3j'))
def __wigner_3j(l1, l2, l3, _version=1):  # pragma: no cover
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
        return torch.einsum('il,jm,kn->ijklmn', (D1, D2, D3)).reshape(n, n)

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
    Q = Q.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

    if next(x for x in Q.flatten() if x.abs() > 1e-10 * Q.abs().max()) < 0:
        Q.neg_()

    Q[Q.abs() < 1e-14] = 0

    with torch_default_dtype(torch.float64):
        abc = rand_angles()
        _Q = torch.einsum("il,jm,kn,lmn", (irr_repr(l1, *abc), irr_repr(l2, *abc), irr_repr(l3, *abc), Q))
        assert torch.allclose(Q, _Q)

    assert Q.dtype == torch.float64
    return Q  # [m1, m2, m3]


################################################################################
# Change of basis
################################################################################

def xyz_to_irreducible_basis(check=True):
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


def irreducible_basis_to_xyz(check=True):
    """
    to convert a vector transforming with irr_repr(1, a, b, c)
    into a vector [x, y, z] transforming with rot(a, b, c)
    see assert for usage

    Inverse of xyz_to_irreducible_basis
    """
    with torch_default_dtype(torch.float64):
        A = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
        if check:
            assert all(torch.allclose(A @ irr_repr(1, a, b, c), rot(a, b, c) @ A) for a, b, c in torch.rand(10, 3))
    return A.type(torch.get_default_dtype())


def xyz3x3_repr(a, b, c):
    """
    representation of 3x3 tensors
    T --> R T R^t
    """
    r = rot(a, b, c)
    return kron(r, r)


def xyz3x3_to_irreducible_basis():
    """
    to convert a 3x3 tensor transforming with xyz3x3_repr(a, b, c)
    into its 1 + 3 + 5 component transforming with irr_repr(0, a, b, c), irr_repr(1, a, b, c), irr_repr(3, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        to1 = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(0, a, b, c) @ to1, to1 @ xyz3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to3 = torch.tensor([
            [0, 0, -1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(1, a, b, c) @ to3, to3 @ xyz3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to5 = torch.tensor([
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [-3**.5 / 3, 0, 0, 0, -3**.5 / 3, 0, 0, 0, 12**.5 / 3],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0]
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(2, a, b, c) @ to5, to5 @ xyz3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

    return to1.type(torch.get_default_dtype()), to3.type(torch.get_default_dtype()), to5.type(torch.get_default_dtype())


def intertwiners(D1, D2, eps=1e-9, with_parity=False):
    """
    Compute a basis of the vector space of matrices A such that
    D1(g) A = A D2(g) for all g in O(3)
    """
    e = (0, 0, 0, 0) if with_parity else (0, 0, 0)
    I1 = D1(*e)
    I2 = D2(*e)

    # picking 20 random rotations seems good enough
    rr = [(rand_angles() + (i % 2,)) if with_parity else rand_angles() for i in range(20)]
    xs = [kron(D1(*g), I2) - kron(I1, D2(*g).T) for g in rr]
    xtx = sum(x.T @ x for x in xs)

    res = xtx.symeig(eigenvectors=True)
    null_space = res.eigenvectors.T[res.eigenvalues.abs() < eps]
    null_space = null_space.reshape(null_space.shape[0], I1.shape[0], I2.shape[0])

    # check that it works
    solutions = []
    for A in null_space:
        d = 0
        for _ in range(4):
            if with_parity:
                r = rand_angles()
                p = torch.randint(0, 2, size=()).item()
                g = r + (p,)
            else:
                g = rand_angles()
            d += A @ D2(*g) - D1(*g) @ A
        d /= 4
        if d.abs().max() < eps:
            solutions.append((d.norm(), A))
    solutions = [A for _, A in sorted(solutions, key=lambda x: x[0])]

    return torch.stack(solutions) if len(solutions) > 0 else torch.zeros(0, I1.shape[0], I2.shape[0])


def reduce(D, D_small, eps=1e-9, with_parity=False):
    """
    Given a "big" representation and a "small" representation
    computes how many times the small appears in the big one and return:
    - how many times the "small" appears in the "big"
    - a matrix that block diagonalize the "big" rep.
    - the remaining of the "big" representation
    """
    def change_and_remove(A, oldD, d):
        def newD(*g):
            return (A @ oldD(*g) @ A.T)[d:][:, d:]
        return newD

    e = (0, 0, 0, 0) if with_parity else (0, 0, 0)
    dim = D(*e).shape[0]
    dim_small = D_small(*e).shape[0]

    D_rest = D
    bigA = torch.eye(dim)
    n = 0

    while True:
        A = intertwiners(D_small, D_rest, eps, with_parity) * dim_small**0.5

        # stops if "small" does not appear in "big" anymore
        if A.shape[0] == 0:
            break

        A, expand = orthonormalize(A[0], eps)
        A = torch.cat([A, expand])

        bigA = direct_sum(torch.eye(n * dim_small), A) @ bigA
        n += 1
        D_rest = change_and_remove(bigA, D, n * dim_small)

    return n, bigA, D_rest


@torch.jit.script
def orthonormalize(
        vecs: torch.Tensor,
        eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    """
    :param vecs: tensor of shape [n, m] with n <= m
    :return: (base, expand)

    base.shape[1] == m
    expand.shape[1] == m
    base.shape[0] + expand.shape[0] == m

    cat[base, expand] is orthonormal
    """
    assert vecs.dim() == 2
    dim = vecs.shape[1]

    base = []
    for x in vecs:
        for y in base:
            x -= torch.dot(x, y) * y
        if x.norm() > 2 * eps:
            x = x / x.norm()
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            base += [x]

    expand = []
    for x in torch.eye(dim, device=vecs.device, dtype=vecs.dtype):
        for y in base + expand:
            x -= torch.dot(x, y) * y
        if x.norm() > 2 * eps:
            x /= x.norm()
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            expand += [x]

    base = torch.stack(base) if len(base) > 0 else vecs.new_zeros(0, dim)
    expand = torch.stack(expand) if len(expand) > 0 else vecs.new_zeros(0, dim)

    return base, expand
