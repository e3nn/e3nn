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

# These functions satisfies that
# rot(*x_to_alpha_beta([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]


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


def test_wigner(l=2):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    a1, b1, c1, a2, b2, c2 = np.random.rand(6)

    r1 = wigner_D_matrix(l, a1, b1, c1)
    r2 = wigner_D_matrix(l, a2, b2, c2)

    a, b, c = compose(a1, b1, c1, a2, b2, c2)
    r = wigner_D_matrix(l, a, b, c)

    r_ = r1 @ r2

    print(np.abs(r - r_).max() / r.std())
