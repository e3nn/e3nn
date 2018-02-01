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
# Analytically derived basis
################################################################################


@cached_dirpklgz("kernels_cache_analytical")
def cube_basis_kernels_analytical(size, n_radial, upsampling, R_out, R_in):
    '''
    Generate equivariant kernel basis mapping between capsules transforming under R_in and R_out
    :param size: side length of the filter kernel (CURRENTLY ONLY ODD SIZES SUPPORTED)
    :param n_radial: number of sampled spherical shells at different radii (CURRENTLY FIXED TO size//2+1)
    :param upsampling: upsampling factor during kernel sampling (NOT IMPLEMENTED YET)
    :param R_out: output representation
    :param R_in: input representation
    :return: basis of equivariant kernels of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
    '''
    # TODO: solve Q_J in subspaces rather than full Q basis
    # TODO: add support for even sidelength kernels
    # TODO: add upsampling (?)

    def _compute_reps_and_basistrafo(R_in, R_out, order_in, order_out):
        '''
        Given the input and output representations compute the change of basis matrix Q
        Q transforms between the space acted on by the tensor product representation and the corresponding space
        acted on by the irrep direct sum representation
        :param R_in: input capsule representation
        :param R_out: output capsule representation
        :param order_in: order of input capsule representation
        :param order_out: order of output capsule representation
        :return: Q basis transform, order of irreps and start indices of irrep blocks
        '''
        from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
        order_irreps = np.arange(abs(order_in-order_out), order_in+order_out+1)
        dim_tensorrep = (2*order_in+1)*(2*order_out+1)
        idxs_start = np.cumsum(np.insert(2*order_irreps+1, 0, 0))
        def _R_tensor(a,b,c):
            ''' tensor representation of R_out and R_in '''
            return np.kron(R_out(a, b, c), R_in(a, b, c))
        def _R_irrep(a,b,c):
            ''' direct sum of irreps representation of R_out and R_in '''
            R_irrep = np.zeros((dim_tensorrep, dim_tensorrep))
            for i in range(len(order_irreps)):
                R_irrep[idxs_start[i]:idxs_start[i+1], idxs_start[i]:idxs_start[i+1]] = wigner_D_matrix(order_irreps[i], a,b,c)
            return R_irrep
        def _sylvester_matrix(a,b,c):
            ''' generate Kronecker product matrix for solving the Sylvester equation '''
            R_tensor = _R_tensor(a,b,c)
            R_irrep  = _R_irrep(a,b,c)
            # inverted wrt notes ( R_tensor = Q R_irrep Q^-1 and K = Q K_tilde )
            return np.kron(np.eye(*R_irrep.shape), R_tensor) - np.kron(R_irrep.T, np.eye(*R_tensor.shape))
            # non-inverted wrt notes ( R_tensor = Q^-1 R_irrep Q and K = Q^-1 K_tilde
            # return np.kron(np.eye(*R_tensor.shape), R_irrep) - np.kron(R_tensor.T, np.eye(*R_irrep.shape))
        def _nullspace(A, eps=1e-13):
            u, s, v = np.linalg.svd(A)#, full_matrices=False)
            null_space = v[s < eps]
            return [b for b in null_space] # list of basis vectors

        N_sample = 5 # number of sampled angles for which the linear system is solved simultaneously
        A_sylvester = np.vstack([_sylvester_matrix(a,b,c) for a,b,c in 2*np.pi*np.random.rand(N_sample,3)])
        # unvectorize
        # transposition is necessary since python uses row major order while vec(Q) is column major
        # optionally leave out transposition and switch Q and Q^-1 (works since reps are orthogonal)
        Q_list = [Q.reshape(dim_tensorrep, dim_tensorrep).T for Q in _nullspace(A_sylvester)]
        # sum up subspace solutions of Q
        Q = np.sum(Q_list, axis=0)
        # check validity of solution
        assert np.allclose(_R_tensor(10, 70, 150), np.dot(np.dot(Q, _R_irrep(10, 70, 150)), np.linalg.inv(Q)))

        #######################################################################################################
        # DEBUG PRINT
        #######################################################################################################
        print('irreps multiplett orders: ', order_irreps, ' dimensionalities: ', 2*order_irreps+1)
        for n,q in enumerate(Q_list):
            print('rank of {}-th subspace solution for Q = {}'.format(n+1, np.linalg.matrix_rank(q)))
        print('rank of summed basis solutions for Q = ', np.linalg.matrix_rank(Q))
        #######################################################################################################

        return Q, order_irreps, idxs_start


    def _sample_basis(size, Q, order_irreps, idxs_start, order_in, order_out):
        '''
        Sample kernels in a cube. Generates radial parts via a fixed heuristic for now.
        :param size: side length of the kernel
        :param Q: change of basis matrix between tensor representation and irrep representation
        :param order_irreps: orders of the irreps in the multiplet
        :param idxs_start: start indices of the irreps blocks
        :param order_in: order of the input representation
        :param order_out: order of the output representation
        :return: sampled equivariant kernel basis of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
        '''
        # sample spherical harmonics on cube, ignoring radial part and aliasing
        from se3_cnn.SO3 import x_to_alpha_beta
        from lie_learn.representations.SO3.spherical_harmonics import sh # real valued by default
        rng = np.linspace(start=-(size // 2), stop=size // 2, num=size, endpoint=True)
        z, y, x = np.meshgrid(rng, rng, rng)
        r_field = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sh_cubes = []
        for idx_J,J in enumerate(order_irreps):
            Y_J_padded = np.zeros((idxs_start[-1], size, size, size))
            for idx_m in range(2*J+1):
                m = idx_m - J
                for idx_x in range(size):
                    for idx_y in range(size):
                        for idx_z in range(size):
                            x = idx_x - size/2 + 0.5
                            y = idx_y - size/2 + 0.5
                            z = idx_z - size/2 + 0.5
                            if x==y==z==0: # angles at origin are nan, special treatment
                                if J==0: # Y^0 is angularly independent, choose any angle
                                    Y_J_padded[idxs_start[idx_J] + idx_m, idx_x, idx_y, idx_z] = sh(J, m, 123, 321)
                                else: # insert zeros for Y^J with J!=0
                                    Y_J_padded[idxs_start[idx_J] + idx_m, idx_x, idx_y, idx_z] = 0
                            else: # not at the origin, sample spherical harmonic
                                alpha, beta = x_to_alpha_beta(np.array([x, y, z]))
                                Y_J_padded[idxs_start[idx_J] + idx_m, idx_x, idx_y, idx_z] = sh(J, m, beta, alpha)

            # ADAPT IF CONVENTION OF Q, Q^-1 IS ALTERED IN _sylvester_matrix ABOVE
            # inverted wrt notes ( R_tensor = Q R_irrep Q^-1 and K = Q K_tilde )
            Y_J_padded = np.einsum('mn,n...->m...', Q, Y_J_padded)
            # non-inverted wrt notes ( R_tensor = Q^-1 R_irrep Q and K = Q^-1 K_tilde
            # Y_J_padded = np.einsum('mn,n...->m...', np.linalg.inv(Q), Y_J_padded)

            # ROW MAJOR VS COLUMN MAJOR (same as in unvec for Q above)
            # gives better results but corresponds to column major
            Y_J_padded = Y_J_padded.reshape(2*order_out+1, 2*order_in+1, size, size, size)
            # row major, should be correct but is not
            # Y_J_padded = Y_J_padded.reshape(2*order_in+1, 2*order_out+1, size, size, size)
            # Y_J_padded = np.transpose(Y_J_padded, axes=(1,0,2,3,4))

            sh_cubes.append(Y_J_padded)

        # WINDOW FUNCTIONS
        # spherical shells with Gaussian radial part
        def _window(r_field, r0, sigma=.6):
            gauss = lambda x, mu, sig: np.exp(-.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
            window = gauss(r_field, r0, sigma)
            size = r_field.shape[0]
            if size % 2 == 1 and r0 != 0:  # angle ambiguity at origin...
                window[size//2, size//2, size//2] = 0
            return window

        # solid ball of radius size//2 (independent of r0)
        # def _window(r_field, r0):
        #     window = (r_field<=(size//2)).astype(int)
        #     window[size//2, size//2, size//2] = 0
        #     return window

        # run over radial parts and window out non-aliased basis functions
        basis = []
        for r,J_max in zip(radii, J_max_list):
            window = _window(r_field, r0=r)
            window = window[np.newaxis,np.newaxis,:]
            for idx_J,J in enumerate(order_irreps):
                if J>J_max:
                    break
                else:
                    basis.append(sh_cubes[idx_J] * window)
        basis = np.stack(basis, axis=0)
        # normalize filter energy
        basis = basis / np.sqrt(np.sum(basis**2, axis=(1,2,3,4,5), keepdims=True))
        return basis


    # feel free to adapt handling of radial part
    assert size % 2 == 1
    assert n_radial == size // 2 + 1  # hardcoded for now
    assert upsampling == 1  # not implemented
    radii = np.arange(n_radial)
    J_max_list = 2 * (radii + 1)
    J_max_list[0] = 0
    # hack to get the orders of the in/out reps
    order_in = (int(R_in.__name__[4:]) - 1) // 2
    order_out = (int(R_out.__name__[4:]) - 1) // 2
    # compute basis
    Q, order_irreps, idxs_start = _compute_reps_and_basistrafo(R_in, R_out, order_in, order_out)
    basis = _sample_basis(size, Q, order_irreps, idxs_start, order_in, order_out)

    #######################################################################################################
    # DEBUG PRINT
    #######################################################################################################
    print('\nkernel size: {}'.format(size))
    print('shell radii: {}'.format(radii))
    print('shell bandlimit: {}'.format(J_max_list))

    print('\ncheck_basis_equivariance for R_in={} -> R_out={}:'.format(R_in.__name__, R_out.__name__))
    accum = np.zeros(len(basis))
    N = 100
    for a,b,c in 2*np.pi*np.random.rand(N,3):
        equiv_vals = check_basis_equivariance(basis, R_out, R_in, a,b,c)
        accum += equiv_vals
    print(accum/N)

    # import se3_cnn.SO3 as SO3
    # reps = [SO3.repr1, SO3.repr3, SO3.repr5]#, SO3.repr7, SO3.repr9, SO3.repr11]
    # for rin in reps:
    #     for rout in reps:
    #         order_in = (int(rin.__name__[4:]) - 1) // 2
    #         order_out = (int(rout.__name__[4:]) - 1) // 2
    #         print('\nBasis for R_in={} -> R_out={}'.format(rin.__name__, rout.__name__))
    #         Q, order_irreps, idxs_start = _compute_reps_and_basistrafo(rin, rout, order_in, order_out)
    #         basis = _sample_basis(size, Q, order_irreps, idxs_start, order_in, order_out)
    #         accum = np.zeros(len(basis))
    #         N = 25
    #         for a, b, c in 2*np.pi*np.random.rand(N, 3):
    #             equiv_vals = check_basis_equivariance(basis, rout, rin, a, b, c)
    #             accum += equiv_vals
    #         accum /= N
    #         print(accum)

    # print('\nCheck basis scalar products for order_in={} -> order_out={}'.format(order_in, order_out))
    # import matplotlib.pyplot as plt
    # overlaps = np.zeros((len(basis), len(basis)))
    # for i,b1 in enumerate(basis):
    #     print('\n')
    #     for j,b2 in enumerate(basis):
    #         overlap = np.sum(b1*b2)
    #         print(i,j, np.round(overlap, decimals=4))
    #         overlaps[i,j] = overlap
    # plt.matshow(overlaps)
    # plt.title('Overlaps between basis elements.')
    # plt.show()
    #######################################################################################################

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
