# pylint: disable=C,E1101
'''
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
'''
import numpy as np


def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])


def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])


def rot(alpha, beta, gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / np.linalg.norm(x)
    beta = np.arccos(x[2])
    alpha = np.arctan2(x[1], x[0])
    return (alpha, beta)


# These functions (x_to_alpha_beta and rot) satisfies that
# rot(*x_to_alpha_beta([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]


def irr_repr(order, alpha, beta, gamma):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    if order == 1:
        # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
        return rot(alpha, beta, gamma)
    return wigner_D_matrix(order, alpha, beta, gamma)


def spherical_harmonics(order, alpha, beta):
    """
    spherical harmonics
    - compatible with irr_repr and compose
    """
    from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default
    Y = np.array([sh(order, m, np.pi - beta, alpha) for m in range(-order, order + 1)])
    if order == 1:
        # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ Y
    return Y


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ np.array([0, 0, 1])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = np.arctan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def _test_irr_repr_are_representation(order):
    """
    This test tests that 
    - irr_repr
    - compose
    are compatible

    D(Z(a1) Y(b1) Z(c1) Z(a2) Y(b2) Z(c2)) = D(Z(a1) Y(b1) Z(c1)) D(Z(a2) Y(b2) Z(c2))
    """
    a1, b1, c1, a2, b2, c2 = np.random.rand(6)

    r1 = irr_repr(order, a1, b1, c1)
    r2 = irr_repr(order, a2, b2, c2)

    a, b, c = compose(a1, b1, c1, a2, b2, c2)
    r = irr_repr(order, a, b, c)

    r_ = r1 @ r2

    d, r = np.abs(r - r_).max(), np.abs(r).max()
    print(d, r)
    assert d < 1e-10 * r


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
    a, b = np.random.rand(2)
    alpha, beta, gamma = np.random.rand(3)

    ra, rb, _ = compose(alpha, beta, gamma, a, b, 0)
    Yrx = spherical_harmonics(order, ra, rb)

    Y = spherical_harmonics(order, a, b)
    DrY = irr_repr(order, alpha, beta, gamma) @ Y

    d, r = np.abs(Yrx - DrY).max(), np.abs(Y).max()
    print(d, r)
    assert d < 1e-10 * r


def _test_change_basis_wigner_to_rot():
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    A = np.array([
        [0, 1, 0],
        [0, 0, 1], 
        [1, 0, 0]
    ])

    a, b, c = np.random.rand(3)

    r1 = A.T @ wigner_D_matrix(1, a, b, c) @ A
    r2 = rot(a, b, c)

    d = np.abs(r1 - r2).max()
    print(d)
    assert d < 1e-10


if __name__ == "__main__":
    print("Change of basis Wigner <-> rot")
    _test_change_basis_wigner_to_rot()
    _test_change_basis_wigner_to_rot()
    _test_change_basis_wigner_to_rot()

    print("Spherical harmonics are solution of Y(rx) = D(r) Y(x)")
    for l in range(7):
        _test_spherical_harmonics(l)

    print("Irreducible repr are indeed representations")
    for l in range(7):
        _test_irr_repr_are_representation(l)
