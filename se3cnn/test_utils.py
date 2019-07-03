from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
import numpy as np


def representation_wigner_matrix(angles, Rs):
    """Transform signal using wigner d matrices for all orders.

    Input:
    - angles (batch, 3)  ZYZ Euler angles
    - signal (batch, channels, point)

    Output: (batch, spectrum_dim, channels)
    """
    sum_dims = np.sum([m * (2 * l + 1) for m, l in Rs])
    rotation = np.zeros([sum_dims, sum_dims])

    start = 0
    for mul, l in Rs:
        wigner = wigner_D_matrix(l, *angles)  # (2 * l + 1) x (2 * l + 1) or mn
        for i in range(mul):
            signal_slice = slice(start, start + (2 * l + 1))
            rotation[signal_slice, signal_slice] = wigner
            start += (2 * l + 1)
    return rotation


def test_identity():
    Rs = [(1, 0), (2, 1), (3, 2)]
    M = representation_wigner_matrix([0, 0, 0], Rs)
    test = np.random.randn(22)
    assert np.allclose(test, np.einsum('mn,n->m', M, test))


def test_transpose():
    Rs = [(1, 0), (2, 1), (3, 2)]
    M = representation_wigner_matrix(np.random.rand(3), Rs)
    test = np.random.randn(22)
    assert np.allclose(test, np.einsum('mn,np,p->m', M.T, M, test))
    assert np.allclose(test, np.einsum('nm,np,p->m', M, M, test))
