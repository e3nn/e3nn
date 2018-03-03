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
# Analytically derived basis
################################################################################

# SHOULD MAYBE NOT BE CACHED BUT ONLY THE EXPENSIVE PART INSIDE (_compute_basistrafo)
# @cached_dirpklgz("kernels_cache_analytical")

def cube_basis_kernels_analytical(size, R_in, R_out, radial_window_dict):
    '''
    Generate equivariant kernel basis mapping between capsules transforming under R_in and R_out
    :param size: side length of the filter kernel (CURRENTLY ONLY ODD SIZES SUPPORTED)
    :param R_out: output representation
    :param R_in: input representation
    :param radial_window_dict: dictionary containing radial_window_fct and its keyword arguments with
                                - radial_window_fct: callable for windowing out radial parts, taking mandatory parameters 'sh_cubes',
                                                     'r_field' and 'order_irreps' as well as optional kwargs via 'radial_window_fct_kwargs'
                                - radial_window_fct_kwargs: keyword arguments for the radial window function
    :return: basis of equivariant kernels of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
    '''
    # TODO: add support for even sidelength kernels
    # TODO: add upsampling (?)

    # @cached_dirpklgz("basis_trafo_generation_cache")
    def _compute_basistrafo(R_in, R_out, order_in, order_out, order_irreps):
        '''
        Given the input and output representations compute the J-subspace change of basis matrices Q_J, each associated with one kernel basis element Y_J
        The direct sum of the Q_J gives the full change of basis matrix Q
        :param R_in: input capsule representation
        :param R_out: output capsule representation
        :param order_in: order of input capsule representation
        :param order_out: order of output capsule representation
        :param order_irreps: orders in the irreps in the multiplet
        :return: list of Q_J basis transforms for subspace J
        '''
        from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

        def _R_tensor(a, b, c): return np.kron(R_out(a, b, c), R_in(a, b, c))

        def _R_irrep_J(J, a, b, c): return wigner_D_matrix(J, a, b, c)

        def _sylvester_submatrix(J, a, b, c):
            ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
            R_tensor = _R_tensor(a, b, c)
            R_irrep_J = _R_irrep_J(J, a, b, c)
            # inverted wrt notes ( R_tensor = Q R_irrep Q^-1 and K = Q K_tilde )
            return np.kron(np.eye(*R_irrep_J.shape), R_tensor) - np.kron(R_irrep_J.T, np.eye(*R_tensor.shape))

        def _nullspace(A, eps=1e-13):
            # sometimes the svd with gesdd does not converge, fall back to gesvd in this case
            try:
                _u, s, v = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesdd')  # pylint: disable=E1123
            except:
                _u, s, v = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')  # pylint: disable=E1123
            null_space = v[s < eps]
            assert null_space.shape[0] == 1  # unique subspace solution
            return null_space[0]

        @cached_dirpklgz("Q_J_cache")
        def _solve_Q_J(J, order_in, order_out, N_sample):
            ''' wrapper to cache expensive solution for J which appears for several j,l combinations
                order_in and order_out are not actually used but the Q_J are of shape ((2*j+1)*(2*l+1), 2*J+1), which means
                that the caching needs to differentiate different input / output orders
             '''
            A_sylvester = np.vstack([_sylvester_submatrix(J, a, b, c) for a, b, c in 2 * np.pi * np.random.rand(N_sample, 3)])
            Q_J = _nullspace(A_sylvester)
            # transposition necessary since 'vec' is defined column major while python is row major
            Q_J = Q_J.reshape(2 * J + 1, (2 * order_in + 1) * (2 * order_out + 1)).T
            assert np.allclose(np.dot(_R_tensor(321, 111, 123), Q_J), np.dot(Q_J, _R_irrep_J(J, 321, 111, 123)))
            return Q_J
        N_sample = 5  # number of sampled angles for which the linear system is solved simultaneously
        Q_list = []
        for J in order_irreps:
            Q_J = _solve_Q_J(J, order_in, order_out, N_sample)
            Q_list.append(Q_J)
        return Q_list

    def _sample_sh_cubes(size, Q_list, order_irreps, order_in, order_out):
        '''
        Sample spherical harmonics in a cube.
        No bandlimiting considered, aliased regions need to be cut by windowing!
        :param size: side length of the kernel
        :param Q: change of basis matrix between tensor representation and irrep representation
        :param order_irreps: orders of the irreps in the multiplet
        :param order_in: order of the input representation
        :param order_out: order of the output representation
        :return: sampled equivariant kernel basis of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
        '''
        # sample spherical harmonics on cube, ignoring radial part and aliasing
        from se3_cnn.SO3 import x_to_alpha_beta
        from lie_learn.representations.SO3.spherical_harmonics import sh  # real valued by default

        def _sample_Y_J(J, r_field):
            ''' sample Y_J on a spatial grid. Returns array of shape (2*J+1, size, size, size) '''
            size = r_field.shape[0]
            Y_J = np.zeros((2 * J + 1, size, size, size))
            for idx_m in range(2 * J + 1):
                m = idx_m - J
                for idx_x in range(size):
                    for idx_y in range(size):
                        for idx_z in range(size):
                            x = idx_x - size / 2 + 0.5
                            y = idx_y - size / 2 + 0.5
                            z = idx_z - size / 2 + 0.5
                            if x == y == z == 0:  # angles at origin are nan, special treatment
                                if J == 0:  # Y^0 is angularly independent, choose any angle
                                    Y_J[idx_m, idx_x, idx_y, idx_z] = sh(0, 0, 123, 321)
                                else:  # insert zeros for Y^J with J!=0
                                    Y_J[idx_m, idx_x, idx_y, idx_z] = 0
                            else:  # not at the origin, sample spherical harmonic
                                alpha, beta = x_to_alpha_beta(np.array([x, y, z]))
                                Y_J[idx_m, idx_x, idx_y, idx_z] = sh(J, m, beta, alpha)
            return Y_J
        rng = np.linspace(start=-(size // 2), stop=size // 2, num=size, endpoint=True)
        z, y, x = np.meshgrid(rng, rng, rng)
        r_field = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sh_cubes = []
        for J, Q_J in zip(order_irreps, Q_list):
            Y_J = _sample_Y_J(J, r_field)
            K_J = np.einsum('mn,n...->m...', Q_J, Y_J)
            K_J = K_J.reshape(2 * order_out + 1, 2 * order_in + 1, size, size, size)
            sh_cubes.append(K_J)
        return sh_cubes, r_field

    # only odd sidelength filters supported for now
    assert size % 2 == 1
    # only irrep representations allowed so far, no tensor representations
    import se3_cnn.SO3 as SO3
    admissible_reps = [SO3.repr1, SO3.repr3, SO3.repr5, SO3.repr7, SO3.repr9, SO3.repr11, SO3.repr13, SO3.repr15]
    assert R_in in admissible_reps and R_out in admissible_reps, 'only irreducible representations allowed with analytical solution, no tensor representations!'
    # hack to get the orders of the in/out reps
    order_in = (dim(R_in) - 1) // 2  # aka j
    order_out = (dim(R_out) - 1) // 2  # aka l
    order_irreps = np.arange(abs(order_in - order_out), order_in + order_out + 1)  # J with |j-l|<=J<=j+l

    # compute basis transformation matrices Q_J
    Q_list = _compute_basistrafo(R_in, R_out, order_in, order_out, order_irreps)
    # sample (basis transformed) spherical harmonics on cube, ignore aliasing
    sh_cubes, r_field = _sample_sh_cubes(size, Q_list, order_irreps, order_in, order_out)
    # window out radial parts
    # make sure to remove aliased regions!
    radial_window_fct = radial_window_dict.get('radial_window_fct')
    radial_window_fct_kwargs = radial_window_dict.get('radial_window_fct_kwargs')
    basis = radial_window_fct(sh_cubes, r_field, order_irreps, **radial_window_fct_kwargs)
    if basis is not None:
        # normalize filter energy (not over axis 0, i.e. different filters are normalized independently)
        basis = basis / np.sqrt(np.sum(basis**2, axis=(1, 2, 3, 4, 5), keepdims=True))
    return basis


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


def gaussian_window_fct_convenience_wrapper(sh_cubes, r_field, order_irreps, mode, border_dist=0., sigma=.6):
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
