# pylint: disable=C,E1101,E1102
'''
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
'''
import torch
import math
from se3cnn.utils import torch_default_dtype
from se3cnn.util.cache_file import cached_dirpklgz


def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return gamma.new_tensor([
        [gamma.cos(), -gamma.sin(), 0],
        [gamma.sin(), gamma.cos(), 0],
        [0, 0, 1]
    ])


def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return beta.new_tensor([
        [beta.cos(), 0, beta.sin()],
        [0, 1, 0],
        [-beta.sin(), 0, beta.cos()]
    ])


def rot(alpha, beta, gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.get_default_dtype())
    x = x / torch.norm(x, 2, -1, keepdim=True)
    beta = torch.acos(x[..., 2])
    alpha = torch.atan2(x[..., 1], x[..., 0])
    return (alpha, beta)  


# These functions (x_to_alpha_beta and rot) satisfies that
# rot(*x_to_alpha_beta([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    # if order == 1:
    #     # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
    #     A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    #     return A @ wigner_D_matrix(1, alpha, beta, gamma) @ A.T
    return torch.tensor(wigner_D_matrix(order, alpha, beta, gamma), dtype=torch.get_default_dtype() if dtype is None else dtype)


# TODO
# vectorize
# order can be a list
# alpha, beta as well
# broadcasting order=int or [],   alpha, beta = float or [, , ]
def spherical_harmonics(order, alpha, beta, dtype=None):
    """
    spherical harmonics
    - compatible with irr_repr and compose
    """
    from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default
    Y = torch.tensor([sh(order, m, math.pi - beta, alpha) for m in range(-order, order + 1)], dtype=torch.get_default_dtype() if dtype is None else dtype)
    # if order == 1:
    #     # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
    #     A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    #     return A @ Y
    return Y


# TODO
# vectorize
# xyz tensor [,,, 3]
def spherical_harmonics_xyz(x, y, z, J):
    with torch_default_dtype(torch.float64):
        if x == y == z == 0:  # angles at origin are nan, special treatment
            if J == 0:  # Y^0 is angularly independent, choose any angle
                return spherical_harmonics(0, 123, 321)  # [m]
            else:  # insert zeros for Y^J with J!=0
                return 0
        else:  # not at the origin, sample spherical harmonic
            alpha, beta = x_to_alpha_beta([x, y, z])
            return spherical_harmonics(J, alpha, beta)  # [m]


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def kron(x, y):
    assert x.ndimension() == 2
    assert y.ndimension() == 2
    return torch.einsum("ij,kl->ikjl", (x, y)).contiguous().view(x.size(0) * y.size(0), x.size(1) * y.size(1))



################################################################################
# Solving the constraint coming from the stabilizer of 0 and e
################################################################################

def get_matrix_kernel(A, eps=1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    _u, s, v = torch.svd(A)

    # A = u @ torch.diag(s) @ v.t()
    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(torch.cat(As, dim=0), eps)


################################################################################
# Analytically derived basis
################################################################################

@cached_dirpklgz("cache/trans_Q")
def basis_transformation_Q_J(J, order_in, order_out, version=3):  # pylint: disable=W0613
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    with torch_default_dtype(torch.float64):
        def _R_tensor(a, b, c): return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

        def _sylvester_submatrix(J, a, b, c):
            ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
            R_tensor = _R_tensor(a, b, c)  # [m_out * m_in, m_out * m_in]
            R_irrep_J = irr_repr(J, a, b, c)  # [m, m]
            return kron(R_tensor, torch.eye(R_irrep_J.size(0))) - \
                kron(torch.eye(R_tensor.size(0)), R_irrep_J.t())  # [(m_out * m_in) * m, (m_out * m_in) * m]

        random_angles = [
            [4.41301023, 5.56684102, 4.59384642],
            [4.93325116, 6.12697327, 4.14574096],
            [0.53878964, 4.09050444, 5.36539036],
            [2.16017393, 3.48835314, 5.55174441],
            [2.52385107, 0.2908958, 3.90040975]
        ]
        null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
        assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
        Q_J = null_space[0]  # [(m_out * m_in) * m]
        Q_J = Q_J.view((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)  # [m_out * m_in, m]
        assert all(torch.allclose(
            _R_tensor(a.item(), b.item(), c.item()) @ Q_J,
            Q_J @ irr_repr(J, a.item(), b.item(), c.item())) for a, b, c in torch.rand(4, 3)
        )

    assert Q_J.dtype == torch.float64
    return Q_J  # [m_out * m_in, m]



################################################################################
# Change of basis
################################################################################


def xyz_vector_basis_to_spherical_basis():
    """
    to convert a vector [x, y, z] transforming with rot(a, b, c)
    into a vector transforming with irr_repr(1, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        A = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)
        assert all(torch.allclose(irr_repr(1, a, b, c) @ A, A @ rot(a, b, c)) for a, b, c in torch.rand(10, 3))
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


################################################################################
# Tests
################################################################################


def test_is_representation(rep):
    """
    rep(Z(a1) Y(b1) Z(c1) Z(a2) Y(b2) Z(c2)) = rep(Z(a1) Y(b1) Z(c1)) rep(Z(a2) Y(b2) Z(c2))
    """
    with torch_default_dtype(torch.float64):
        a1, b1, c1, a2, b2, c2 = torch.rand(6)

        r1 = rep(a1, b1, c1)
        r2 = rep(a2, b2, c2)

        a, b, c = compose(a1, b1, c1, a2, b2, c2)
        r = rep(a, b, c)

        r_ = r1 @ r2

        d, r = (r - r_).abs().max(), r.abs().max()
        print(d.item(), r.item())
        assert d < 1e-10 * r, d / r


def _test_spherical_harmonics(order):
    """
    This test tests that
    - irr_repr
    - compose
    - spherical_harmonics
    are compatible

    Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
    with x = Z(a) Y(b) eta
    """
    with torch_default_dtype(torch.float64):
        a, b = torch.rand(2)
        alpha, beta, gamma = torch.rand(3)

        ra, rb, _ = compose(alpha, beta, gamma, a, b, 0)
        Yrx = spherical_harmonics(order, ra, rb)

        Y = spherical_harmonics(order, a, b)
        DrY = irr_repr(order, alpha, beta, gamma) @ Y

        d, r = (Yrx - DrY).abs().max(), Y.abs().max()
        print(d.item(), r.item())
        assert d < 1e-10 * r, d / r


def _test_change_basis_wigner_to_rot():
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    with torch_default_dtype(torch.float64):
        A = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=torch.float64)

        a, b, c = torch.rand(3)

        r1 = A.t() @ torch.tensor(wigner_D_matrix(1, a, b, c), dtype=torch.float64) @ A
        r2 = rot(a, b, c)

        d = (r1 - r2).abs().max()
        print(d.item())
        assert d < 1e-10


if __name__ == "__main__":
    from functools import partial

    print("Change of basis")
    xyz_vector_basis_to_spherical_basis()
    test_is_representation(tensor3x3_repr)
    tensor3x3_repr_basis_to_spherical_basis()

    print("Change of basis Wigner <-> rot")
    _test_change_basis_wigner_to_rot()
    _test_change_basis_wigner_to_rot()
    _test_change_basis_wigner_to_rot()

    print("Spherical harmonics are solution of Y(rx) = D(r) Y(x)")
    for l in range(7):
        _test_spherical_harmonics(l)

    print("Irreducible repr are indeed representations")
    for l in range(7):
        test_is_representation(partial(irr_repr, l))
