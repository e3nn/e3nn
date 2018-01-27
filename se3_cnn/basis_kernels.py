# pylint: disable=C,R,E1101
'''
Given two representation of SO(3), computes the basis elements of
the vector space of kernels K such that
    integral dy K(x, y) f(y)
is equivariant.

K must satifies
    K(ux, uy) = R_out(u) K(x, y) R_in(u^{-1}) for all u in SE(3)

Therefore
    K(x, y) = K(0, y-x)

    K(0, x) = K(0, g |x| e)  where e is a prefered chosen unit vector and g is in SO(3)
'''
import numpy as np
from se3_cnn.util.cache_file import cached_dirpklgz
from se3_cnn.SO3 import dim

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

    >>> A = np.array([[1, -1], [-1, 1]])
    >>> ks = get_matrix_kernel(A)
    >>> np.linalg.norm(ks[0] - np.array([1, 1]) / np.sqrt(2)) < 1e-10
    True
    '''
    u, s, v = np.linalg.svd(A, full_matrices=False)  # pylint: disable=W0612
    # A = u @ np.diag(s) @ v
    kernel = v[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(np.concatenate(As, axis=0), eps)


def basis_kernels_satisfying_Zrot_constraint(R_out, R_in):
    '''
    :return: list of dim(R_out) x dim(R_in) matrices

    Computes a basis of the vector space of matrices K such that
        R_out(g) K = K R_in(g) for all g in Z rotations (stabilizer of 0 and ez)

    The following is useful to understand how the kroneker product is used
        R_ij K_jk = K_ij R_jk
        R_ij K_jk - K_ij R_jk = 0
        (R_ix delta_ky - delta_ix R_yk) K_xy = 0
        (kron(R,1)_(ik)(xy) - kron(1,R.T)_(ik)(xy)) K_(xy) = 0
    We can see the kroneker product simply as regrouping two indices in one
    '''
    def kron(gamma, R_out, R_in):
        return np.kron(R_out(0, 0, gamma), np.eye(dim(R_in))) - np.kron(np.eye(dim(R_out)), R_in(0, 0, gamma).T)

    some_random_angles = [np.pi, 1, np.pi / 7, 1.54321]
    As = [kron(gamma, R_out, R_in) for gamma in some_random_angles]
    kA = get_matrices_kernel(As, 1e-10)

    # K_(xy) --> K_xy
    basis_elements = kA.reshape((-1, dim(R_out), dim(R_in)))
    #basis_elements = [x.reshape((dim(R_out), dim(R_in))) for x in kA]

    def check(K, gamma):
        '''
        Check that K satifies R_out K = K R_in
        '''
        return np.linalg.norm(np.dot(R_out(0, 0, gamma), K) - np.dot(K, R_in(0, 0, gamma))) < 1e-10

    assert all([check(K, gamma) for K in basis_elements for gamma in [0, np.pi / 4, np.pi / 2, np.random.rand(), np.random.rand()]])

    return basis_elements


def basis_kernels_satisfying_SO3_constraint(R_out, R_in):
    '''
    :return: list of dim(R_out) x dim(R_in) matrices

    Computes a basis of the vector space of matrices K such that
        R_out(g) K = K R_in(g) for all g in SO(3) (stabilizer of 0)

    The following is useful to understand how the kroneker product is used
        R_ij K_jk = K_ij R_jk
        R_ij K_jk - K_ij R_jk = 0
        (R_ix delta_ky - delta_ix R_yk) K_xy = 0
        (kron(R,1)_(ik)(xy) - kron(1,R.T)_(ik)(xy)) K_(xy) = 0
    We can see the kroneker product simply as regrouping two indices in one
    '''
    def kron(alpha, beta, gamma, R_out, R_in):
        return np.kron(R_out(alpha, beta, gamma), np.eye(dim(R_in))) - np.kron(np.eye(dim(R_out)), R_in(alpha, beta, gamma).T)

    some_random_alpha = [np.pi, 1, np.pi / 7, 1.54321]
    some_random_beta = [1, np.pi / 7, 1.32]
    some_random_gamma = [np.pi, 1, np.pi / 8, 1.324]
    As = [kron(alpha, beta, gamma, R_out, R_in) for alpha in some_random_alpha for beta in some_random_beta for gamma in some_random_gamma]
    kA = get_matrices_kernel(As, 1e-10)

    # K_(xy) --> K_xy
    basis_elements = kA.reshape((-1, dim(R_out), dim(R_in)))
    #basis_elements = [x.reshape((dim(R_out), dim(R_in))) for x in kA]

    def check(K, alpha, beta, gamma):
        '''
        Check that K satifies R_out K = K R_in
        '''
        return np.linalg.norm(np.dot(R_out(alpha, beta, gamma), K) - np.dot(K, R_in(alpha, beta, gamma))) < 1e-10

    assert all([check(K, alpha, beta, gamma) for K in basis_elements
                for alpha in [0, np.pi / 4, np.pi / 2, np.random.rand(), np.random.rand()]
                for beta in [1, 2]
                for gamma in [np.random.rand(), -np.random.rand()]])

    return basis_elements

################################################################################
# Constructing kernel basis elements
################################################################################


def transport_kernel(x, base0e, R_out, R_in):
    '''
    "Transport" the kernel K(0, ez) to K(0, x)

    K(0, x) = K(0, g |x| ez) = R_out(g) K(0, |x| ez) R_in(g)^{-1}

    In this function: K(0, |x| ez) = K(0, ez)
    '''
    from se3_cnn.SO3 import x_to_alpha_beta
    alpha, beta = x_to_alpha_beta(x)
    # inv(R_in(alpha, beta, 0)) = inv(R_in(Z(alpha) Y(beta))) = R_in(Y(-beta) Z(-alpha))
    return np.matmul(np.matmul(R_out(alpha, beta, 0), base0e), R_in(0, -beta, -alpha))


def cube_basis_kernels(size, R_out, R_in):
    dim_in = dim(R_in)
    dim_out = dim(R_out)

    # compute the basis of K(0, ez)
    basis = basis_kernels_satisfying_Zrot_constraint(R_out, R_in)

    result = np.empty((len(basis), dim_out, dim_in, size, size, size))

    for xi in range(size):
        for yi in range(size):
            for zi in range(size):
                x = xi - size / 2 + 0.5
                y = yi - size / 2 + 0.5
                z = zi - size / 2 + 0.5
                point = np.array([x, y, z])

                if x == 0 and y == 0 and z == 0:
                    result[:, :, :, xi, yi, zi] = 0
                else:
                    result[:, :, :, xi, yi, zi] = transport_kernel(point, basis, R_out, R_in)
    return result

################################################################################
# Subsampling function
################################################################################


def gaussian_subsampling(im, M):
    '''
    :param im: array of dimentions (d0, d1, d2, ...)
    :return: array of dimentions (d0 / M[0], d1 / M[1], d2 / M[2], ...)
    '''
    import scipy.ndimage
    M = np.array(M)
    assert M.dtype == np.int
    assert np.all(M % 2 == 1)

    sigma = 0.5 * np.sqrt(M**2 - 1)
    im = scipy.ndimage.filters.gaussian_filter(im, sigma, mode='constant')

    s = [slice(m // 2, None, m) for m in M]
    return im[s]


################################################################################
# Orthonormalization
################################################################################
def orthonormalize(basis):
    import scipy.linalg

    shape = basis.shape
    basis = basis.reshape((shape[0], -1))

    basis = scipy.linalg.orth(basis.T).T

    basis = basis.reshape((-1,) + shape[1:])
    return basis

################################################################################
# Full generation
################################################################################


@cached_dirpklgz("kernels_cache_hat")
def cube_basis_kernels_subsampled_hat(size, n_radial, upsampling, R_out, R_in):
    basis = cube_basis_kernels(size * upsampling, R_out, R_in)
    rng = np.linspace(start=-1, stop=1, num=size * upsampling, endpoint=True)
    z, y, x = np.meshgrid(rng, rng, rng)
    r = np.sqrt(x**2 + y**2 + z**2)

    kernels = []

    step = 1 / n_radial
    w = 0.5 / n_radial
    for i in range(0, n_radial):
        c = step * i + w
        mask = w - np.abs(r - c)
        mask[r > c + w] = 0
        mask[r < c - w] = 0

        kernels.append(basis * mask)
    basis = np.concatenate(kernels)

    basis = orthonormalize(basis)

    return gaussian_subsampling(basis, (1, 1, 1, upsampling, upsampling, upsampling))


################################################################################
# Measure equivariance
################################################################################

def check_basis_equivariance(basis, R_out, R_in, alpha, beta, gamma):
    from se3_cnn import SO3
    from scipy.ndimage import affine_transform

    n = basis.shape[0]
    dim_in = SO3.dim(R_in)
    dim_out = SO3.dim(R_out)
    size = basis.shape[-1]
    assert basis.shape == (n, dim_out, dim_in, size, size, size), basis.shape

    basis = basis / np.linalg.norm(basis.reshape((n, -1)), axis=1).reshape((-1, 1, 1, 1, 1, 1))

    x = basis.reshape((-1, size, size, size))
    y = np.empty_like(x)

    invrot = SO3.rot(-gamma, -beta, -alpha)
    center = (np.array(x.shape[1:]) - 1) / 2

    for k in range(y.shape[0]):
        y[k] = affine_transform(x[k], matrix=invrot, offset=center - np.dot(invrot, center))

    y = y.reshape(basis.shape)

    y = np.einsum("ij,bjk...,kl->bil...", R_out(alpha, beta, gamma), y, R_in(-gamma, -beta, -alpha))

    return np.array([np.sum(basis[i] * y[i]) for i in range(n)])


################################################################################
# Testing
################################################################################
if __name__ == '__main__':
    import doctest
    doctest.testmod()
