# pylint: disable=C,R,E1101
import numpy as np
import scipy.linalg
import scipy.ndimage
from se3_cnn.util.cache_file import cached_dirpklgz
from se3_cnn.SO3 import dim
from se3_cnn import SO3
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def cube_basis_kernels_analytical(R_in, R_out, beta, alpha):
    '''
    Generate equivariant kernel basis mapping between capsules transforming under R_in and R_out
    :param size: side length of the filter kernel (CURRENTLY ONLY ODD SIZES SUPPORTED)
    :param R_out: output representation
    :param R_in: input representation
    :param radial_window: callable for windowing out radial parts, taking mandatory parameters
                          'sh_cubes', 'r_field' and 'order_irreps'
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

    def _sample_sh_cubes(beta, alpha, Q_list, order_irreps, order_in, order_out):
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

        def _sample_Y_J(J, beta, alpha):
            ''' sample Y_J on a spatial grid. Returns array of shape (2*J+1, size, size, size) '''
            assert len(beta) == len(alpha)
            #size = r_field.shape[0]
            Y_J = np.zeros((2 * J + 1, len(beta)))
            for idx_m in range(2 * J + 1):
                m = idx_m - J
                for idx, (b, a) in enumerate(zip(beta, alpha)):
                    Y_J[idx_m, idx] = sh(J, m, b, a)
            return Y_J
        #rng = np.linspace(start=-(size // 2), stop=size // 2, num=size, endpoint=True)
        #z, y, x = np.meshgrid(rng, rng, rng)
        #r_field = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sh_cubes = []
        for J, Q_J in zip(order_irreps, Q_list):
            Y_J = _sample_Y_J(J, beta, alpha)
            K_J = np.einsum('mn,n...->m...', Q_J, Y_J)
            K_J = K_J.reshape(2 * order_out + 1, 2 * order_in + 1, -1)
            sh_cubes.append(K_J)
        return sh_cubes

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
    sh_cubes = _sample_sh_cubes(beta, alpha, Q_list, order_irreps, order_in, order_out)
    return sh_cubes


def plot_sphere(beta, alpha, f):
    alpha = np.concatenate((alpha, alpha[:, :1]), axis=1)
    beta = np.concatenate((beta, beta[:, :1]), axis=1)
    f = np.concatenate((f, f[:, :1]), axis=1)

    x = np.sin(beta) * np.cos(alpha)
    y = np.sin(beta) * np.sin(alpha)
    z = np.cos(beta)

    fc = cm.gray(f)
    fc = plt.get_cmap("bwr")(f)

    #fig = plt.figure(figsize=(5, 3))
    #ax = fig.add_subplot(111, projection='3d', aspect=1)
    ax = plt.gca()
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=fc)  # cm.gray(f))
    # Turn off the axis planes
    ax.view_init(azim=0, elev=90)
    ax.set_axis_off()
    a = 0.6
    ax.set_xlim3d(-a, a)
    ax.set_ylim3d(-a, a)
    ax.set_zlim3d(-a, a)


def main():
    b = 25
    beta = np.linspace(0, np.pi / 2, 2 * b)
    alpha = np.arange(2 * b) * np.pi / b
    beta, alpha = np.meshgrid(beta, alpha, indexing='ij')
    f = cube_basis_kernels_analytical(SO3.repr3, SO3.repr3, beta.flatten(), alpha.flatten() + np.pi / b / 2)
    f = np.array(f)
    f = (f - np.min(f)) / (np.max(f) - np.min(f))

    f = f.reshape(*f.shape[:3], *beta.shape)

    nbase = f.shape[0]
    dim_out = f.shape[1]
    dim_in = f.shape[2]

    w = 1
    fig = plt.figure(figsize=(nbase * dim_in + (nbase - 1) * w, dim_out))

    for base in range(nbase):
        for i in range(dim_out):
            for j in range(dim_in):
                width = 1 / (nbase * dim_in + (nbase - 1) * w)
                height = 1 / dim_out
                rect = [
                    (base * (dim_in + w) + j) * width,
                    (dim_out - i - 1) * height,
                    width,
                    height
                ]
                fig.add_axes(rect, projection='3d', aspect=1)
                plot_sphere(beta, alpha, f[base, i, j])

    plt.savefig("kernels.png")


main()
