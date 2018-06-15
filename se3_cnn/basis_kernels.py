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
import scipy.linalg
import scipy.ndimage
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
from se3_cnn.SO3 import x_to_alpha_beta
from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default
from se3_cnn.util.cache_file import cached_dirpklgz


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
    >>> np.linalg.norm(ks[0] - np.array([1, 1]) / np.sqrt(2)) < 1e-10 or np.linalg.norm(ks[0] + np.array([1, 1]) / np.sqrt(2)) < 1e-10
    True
    '''
    try:
        _u, s, v = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesdd')  # pylint: disable=E1123
    except:  # pylint: disable=W
        _u, s, v = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')  # pylint: disable=E1123

    # A = u @ np.diag(s) @ v
    kernel = v[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(np.concatenate(As, axis=0), eps)


################################################################################
# Analytically derived basis
################################################################################

@cached_dirpklgz("cache/trans_Q")
def _basis_transformation_Q_J(J, order_in, order_out):
    def _R_tensor(a, b, c): return np.kron(wigner_D_matrix(order_out, a, b, c), wigner_D_matrix(order_in, a, b, c))

    def _sylvester_submatrix(J, a, b, c):
        ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
        R_tensor = _R_tensor(a, b, c)
        R_irrep_J = wigner_D_matrix(J, a, b, c)
        # inverted wrt notes ( R_tensor = Q R_irrep Q^-1 and K = Q K_tilde )
        return np.kron(np.eye(*R_irrep_J.shape), R_tensor) - np.kron(R_irrep_J.T, np.eye(*R_tensor.shape))

    random_angles = [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.2908958, 3.90040975]
    ]
    null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
    assert null_space.shape[0] == 1  # unique subspace solution
    Q_J = null_space[0]
    # transposition necessary since 'vec' is defined column major while python is row major
    Q_J = Q_J.reshape(2 * J + 1, (2 * order_in + 1) * (2 * order_out + 1)).T
    assert np.allclose(np.dot(_R_tensor(321, 111, 123), Q_J), np.dot(Q_J, wigner_D_matrix(J, 321, 111, 123)))
    return Q_J


@cached_dirpklgz("cache/sh_cube")
def _sample_sh_cube(size, J):
    '''
    Sample spherical harmonics in a cube.
    No bandlimiting considered, aliased regions need to be cut by windowing!
    :param size: side length of the kernel
    :param J: order of the spherical harmonics
    '''
    rng = np.linspace(start=-((size - 1) / 2), stop=(size - 1) / 2, num=size, endpoint=True)

    Y_J = np.zeros((2 * J + 1, size, size, size))
    for idx_m in range(2 * J + 1):
        m = idx_m - J
        for idx_z, z in enumerate(rng):
            for idx_y, y in enumerate(rng):
                for idx_x, x in enumerate(rng):
                    if x == y == z == 0:  # angles at origin are nan, special treatment
                        if J == 0:  # Y^0 is angularly independent, choose any angle
                            Y_J[idx_m, idx_z, idx_y, idx_x] = sh(0, 0, 123, 321)
                        else:  # insert zeros for Y^J with J!=0
                            Y_J[idx_m, idx_z, idx_y, idx_x] = 0
                    else:  # not at the origin, sample spherical harmonic
                        # To end up with the convention : vector_field[z, y, x] = np.array([v_x, v_y, v_z])
                        # Instead of x_to_alpha_beta(np.array([x, y, z]))
                        # We need to do this (trust me)
                        alpha, beta = x_to_alpha_beta(np.array([-z, -x, y]))
                        Y_J[idx_m, idx_z, idx_y, idx_x] = sh(J, m, beta, alpha)

    return Y_J


def _sample_cube(size, order_in, order_out):
    '''
    Sample spherical harmonics in a cube.
    No bandlimiting considered, aliased regions need to be cut by windowing!
    :param size: side length of the kernel
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: sampled equivariant kernel basis of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
    '''

    rng = np.linspace(start=-((size - 1) / 2), stop=(size - 1) / 2, num=size, endpoint=True)

    order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
    sh_cubes = []
    for J in order_irreps:
        Y_J = _sample_sh_cube(size, J)

        # compute basis transformation matrix Q_J
        Q_J = _basis_transformation_Q_J(J, order_in, order_out)
        K_J = np.einsum('mn,n...->m...', Q_J, Y_J)
        K_J = K_J.reshape(2 * order_out + 1, 2 * order_in + 1, size, size, size)
        sh_cubes.append(K_J)

    z, y, x = np.meshgrid(rng, rng, rng)
    r_field = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return sh_cubes, r_field, order_irreps


def cube_basis_kernels_analytical(size, order_in, order_out, radial_window):
    '''
    Generate equivariant kernel basis mapping between capsules transforming under order_in and order_out
    :param size: side length of the filter kernel
    :param order_out: output representation order
    :param order_in: input representation order
    :param radial_window: callable for windowing out radial parts, taking mandatory parameters
                          'sh_cubes', 'r_field' and 'order_irreps'
    :return: basis of equivariant kernels of shape (N_basis, 2 * order_out + 1, 2 * order_in + 1, size, size, size)
    '''
    # TODO: add upsampling (?)

    # sample (basis transformed) spherical harmonics on cube, ignore aliasing
    # window out radial parts
    # make sure to remove aliased regions!
    basis = radial_window(*_sample_cube(size, order_in, order_out))
    if basis is not None:
        # normalize filter energy (not over axis 0, i.e. different filters are normalized independently)
        basis = basis / np.sqrt(np.sum(basis ** 2, axis=(1, 2, 3, 4, 5), keepdims=True))
    return basis


################################################################################
# Radial distribution functions
################################################################################

def gaussian_window_fct(sh_cubes, r_field, order_irreps, radii, J_max_list, sigma=.6):
    '''
    gaussian windowing function with manual handling of shell radii, shell bandlimits and shell width
    :param sh_cubes: list of spherical harmonic basis cubes which are np.ndarrays of shape (2*l+1, 2*j+1, size, size, size)
    :param r_field: np.ndarray containing radial coordinates in the cube, shape (size,size,size)
    :param order_irreps: np.ndarray with the order J of the irreps in sh_cubes with |j-l|<=J<=j+l
    :param radii: np.ndarray with radii of the shells, sets mean of the radial gaussians
    :param J_max_list: np.ndarray with bandlimits of the shells, same length as radii
    :param sigma: width of the shells, corresponds to standard deviation of radial gaussians
    '''
    # spherical shells with Gaussian radial part
    def _gauss_window(r_field, r0, sigma):
        return np.exp(-.5 * ((r_field - r0) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    # run over radial parts and window out non-aliased basis functions
    assert len(radii) == len(J_max_list)
    basis = []
    for r, J_max in zip(radii, J_max_list):
        window = _gauss_window(r_field, r0=r, sigma=sigma)
        window = window[np.newaxis, np.newaxis, :]
        # for each spherical shell at radius r window sh_cube if J does not exceed the bandlimit J_max
        for idx_J, J in enumerate(order_irreps):
            if J > J_max:
                break
            else:
                basis.append(sh_cubes[idx_J] * window)
    if len(basis) > 0:
        basis = np.stack(basis, axis=0)
    else:
        basis = None
    return basis


def gaussian_window_fct_convenience_wrapper(sh_cubes, r_field, order_irreps, mode='compromise', border_dist=0., sigma=.6):
    '''
    convenience wrapper for windowing function with three different predefined modes for radii and bandlimits
    :param sh_cubes: list of spherical harmonic basis cubes which are np.ndarrays of shape (2*l+1, 2*j+1, size, size, size)
    :param r_field: np.ndarray containing radial coordinates in the cube, shape (size,size,size)
    :param order_irreps: np.ndarray with the order J of the irreps in sh_cubes with |j-l|<=J<=j+l
    :param mode: string in ['sfcnn', 'conservative', '']
                 'conservative': radial bandlimits such that equivariance value stays over 90% (for border_dist=0 and except for outer shell)
                          [0,2,4,6,8,10,...]
                 'compromise' something inbetween
                          [0,3,5,7,9,11,...]
                 'sfcnn': same radial bandlimits as used in https://arxiv.org/abs/1711.07289
                          [0,4,6,8,10,12,...]
    :param border_dist: distance of mean of outermost shell from outermost pixel center
    :param sigma: width of the shell
    '''
    assert mode in ['conservative', 'compromise', 'sfcnn']
    size = r_field.shape[0]
    # radii = np.arange(size//2 + 1)
    n_radial = size // 2 + 1
    radii = np.linspace(start=0, stop=size // 2 - border_dist, num=n_radial)
    if mode == 'conservative':
        J_max_list = np.array([0, 2, 4, 6, 8, 10, 12, 14])[:n_radial]
    if mode == 'compromise':
        J_max_list = np.array([0, 3, 5, 7, 9, 11, 13, 15])[:n_radial]
    if mode == 'sfcnn':
        # J_max_list = np.floor(2*(radii + 1))
        # J_max_list[0] = 0
        J_max_list = np.array([0, 4, 6, 8, 10, 12, 14, 16])[:n_radial]
    basis = gaussian_window_fct(sh_cubes, r_field, order_irreps, radii, J_max_list, sigma)
    return basis


################################################################################
# Measure equivariance
################################################################################

def check_basis_equivariance(basis, order_in, order_out, alpha, beta, gamma):
    from se3_cnn import SO3
    from scipy.ndimage import affine_transform

    n = basis.shape[0]
    dim_in = 2 * order_in + 1
    dim_out = 2 * order_out + 1
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

    y = np.einsum("ij,bjk...,kl->bil...", wigner_D_matrix(order_out, alpha, beta, gamma), y, wigner_D_matrix(order_in, -gamma, -beta, -alpha))

    return np.array([np.sum(basis[i] * y[i]) for i in range(n)])


################################################################################
# Testing
################################################################################

if __name__ == '__main__':
    import doctest
    doctest.testmod()
