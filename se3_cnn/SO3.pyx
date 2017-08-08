# pylint: disable=C,E1101
'''
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
'''
import numpy as np
cimport numpy as np


cpdef rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])


cpdef rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                     [0, 1, 0],
                     [-np.sin(beta), 0, np.cos(beta)]])


cpdef rot(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha).dot(rot_y(beta)).dot(rot_z(gamma))


cpdef x_to_alpha_beta(np.ndarray[np.float64_t, ndim=1] x):
    x = x / np.linalg.norm(x)
    cdef np.float64_t beta = np.arccos(x[2])
    cdef np.float64_t alpha = np.arctan2(x[1], x[0])
    return (alpha, beta)


cpdef dim(R):
    return R(0, 0, 0).shape[0]

# The next functions are some usual representations


cpdef scalar_repr(np.float64_t alpha, np.float64_t beta, np.float64_t gamma): # pylint: disable=W0613
    return np.array([[1]])

cpdef vector_repr(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return rot(alpha, beta, gamma)

cpdef tensor_repr(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    r = vector_repr(alpha, beta, gamma)
    return np.kron(r, r)


from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

cpdef repr1(np.float64_t alpha, np.float64_t beta, np.float64_t gamma): # pylint: disable=W0613
    return np.eye(1)

cpdef repr3(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return wigner_D_matrix(1, alpha, beta, gamma)

cpdef repr5(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return wigner_D_matrix(2, alpha, beta, gamma)

cpdef repr7(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return wigner_D_matrix(3, alpha, beta, gamma)

cpdef repr9(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return wigner_D_matrix(4, alpha, beta, gamma)

cpdef repr11(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    return wigner_D_matrix(5, alpha, beta, gamma)

cpdef repr3x3(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    r = repr3(alpha, beta, gamma)
    return np.kron(r, r)

cpdef repr3x5(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    r1 = repr3(alpha, beta, gamma)
    r2 = repr5(alpha, beta, gamma)
    return np.kron(r1, r2)

cpdef repr5x3(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    r1 = repr5(alpha, beta, gamma)
    r2 = repr3(alpha, beta, gamma)
    return np.kron(r1, r2)

cpdef repr5x5(np.float64_t alpha, np.float64_t beta, np.float64_t gamma):
    r = repr5(alpha, beta, gamma)
    return np.kron(r, r)
