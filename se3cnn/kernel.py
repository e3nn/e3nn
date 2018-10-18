# pylint: disable=C,R,E1101,E1102
'''
Given two representation of SO(3), computes the basis elements of
the vector space of kernels K such that
    integral dy K(x, y) f(y)
is equivariant.

K must satifies
    K(ux, uy) = R_out(u) K(x, y) R_in(u^{-1}) for all u in SE(3)

Therefore
    K(x, y) = K(0, y-x)

    K(0, g |x| e) = R_out(g) K(0, |x| e) R_in(g^{-1})  where e is a prefered chosen unit vector and g is in SO(3)
'''
import torch
from se3cnn.SO3 import x_to_alpha_beta, irr_repr, spherical_harmonics, kron, torch_default_dtype
from se3cnn.util.cache_file import cached_dirpklgz
import math


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
def _basis_transformation_Q_J(J, order_in, order_out, version=3):  # pylint: disable=W0613
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


@cached_dirpklgz("cache/sh_cube")
def _sample_sh_cube(size, J, version=3):  # pylint: disable=W0613
    '''
    Sample spherical harmonics in a cube.
    No bandlimiting considered, aliased regions need to be cut by windowing!
    :param size: side length of the kernel
    :param J: order of the spherical harmonics
    '''
    with torch_default_dtype(torch.float64):
        rng = torch.linspace(-((size - 1) / 2), (size - 1) / 2, steps=size)

        Y_J = torch.zeros(2 * J + 1, size, size, size, dtype=torch.float64)
        for idx_x, x in enumerate(rng):
            for idx_y, y in enumerate(rng):
                for idx_z, z in enumerate(rng):
                    if x == y == z == 0:  # angles at origin are nan, special treatment
                        if J == 0:  # Y^0 is angularly independent, choose any angle
                            Y_J[:, idx_x, idx_y, idx_z] = spherical_harmonics(0, 123, 321)  # [m]
                        else:  # insert zeros for Y^J with J!=0
                            Y_J[:, idx_x, idx_y, idx_z] = 0
                    else:  # not at the origin, sample spherical harmonic
                        alpha, beta = x_to_alpha_beta([x, y, z])
                        Y_J[:, idx_x, idx_y, idx_z] = spherical_harmonics(J, alpha, beta)  # [m]

    assert Y_J.dtype == torch.float64
    return Y_J  # [m, x, y, z]


def _sample_cube(size, order_in, order_out):
    '''
    :param size: side length of the kernel
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: sampled equivariant kernel basis of shape (N_basis, 2*order_out+1, 2*order_in+1, size, size, size)
    '''

    rng = torch.linspace(-((size - 1) / 2), (size - 1) / 2, steps=size, dtype=torch.float64)

    order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
    solutions = []
    for J in order_irreps:
        Y_J = _sample_sh_cube(size, J)  # [m, x, y, z]

        # compute basis transformation matrix Q_J
        Q_J = _basis_transformation_Q_J(J, order_in, order_out)  # [m_out * m_in, m]
        K_J = torch.einsum('mn,nxyz->mxyz', (Q_J, Y_J))  # [m_out * m_in, x, y, z]
        K_J = K_J.view(2 * order_out + 1, 2 * order_in + 1, size, size, size)  # [m_out, m_in, x, y, z]
        solutions.append(K_J)

        # check that  rho_out(u) K(u^-1 x) rho_in(u^-1) = K(x) with u = rotation of +pi/2 around y axis
        tmp = K_J.transpose(2, 4).flip(4)  # K(u^-1 x)
        tmp = torch.einsum(
            "ij,jkxyz,kl->ilxyz",
            (
                irr_repr(order_out, 0, math.pi / 2, 0, dtype=K_J.dtype),
                tmp,
                irr_repr(order_in, 0, -math.pi / 2, 0, dtype=K_J.dtype)
            )
        )  # rho_out(u) K(u^-1 x) rho_in(u^-1)
        assert torch.allclose(tmp, K_J)

    r_field = (rng.view(-1, 1, 1).pow(2) + rng.view(1, -1, 1).pow(2) + rng.view(1, 1, -1).pow(2)).sqrt()  # [x, y, z]
    return solutions, r_field, order_irreps


def cube_basis_kernels(size, order_in, order_out, radial_window):
    '''
    Generate equivariant kernel basis mapping between capsules transforming under order_in and order_out
    :param size: side length of the filter kernel
    :param order_in: input representation order
    :param order_out: output representation order
    :param radial_window: callable for windowing out radial parts, taking mandatory parameters 'solutions', 'r_field' and 'order_irreps'
    :return: basis of equivariant kernels of shape (N_basis, 2 * order_out + 1, 2 * order_in + 1, size, size, size)
    '''
    basis = radial_window(*_sample_cube(size, order_in, order_out))
    if basis is None:
        return None

    # check that  rho_out(u) K(u^-1 x) rho_in(u^-1) = K(x) with u = rotation of +pi/2 around y axis
    tmp = basis.transpose(3, 5).flip(5)  # K(u^-1 x)
    tmp = torch.einsum(
        "ij,bjkxyz,kl->bilxyz",
        (
            irr_repr(order_out, 0, math.pi / 2, 0, dtype=basis.dtype),
            tmp,
            irr_repr(order_in, 0, -math.pi / 2, 0, dtype=basis.dtype)
        )
    )  # rho_out(u) K(u^-1 x) rho_in(u^-1)
    assert torch.allclose(tmp, basis)

    return basis


################################################################################
# Radial distribution functions
################################################################################

def sigmoid_window(solutions, r_field, order_irreps, sharpness=5):
    '''
    sigmoid windowing function
    takes as input the output of _sample_cube
    '''
    size = r_field.size(0)  # 5
    n_radial = size // 2 + 1  # 3
    radii = torch.linspace(0.5, size // 2 + 0.5, steps=n_radial, dtype=torch.float64)  # [0.5, 1.5, 2.5]

    basis = []
    for i, r in enumerate(radii):
        window = torch.sigmoid(sharpness * (r - r_field))
        J_max = 2 * i + 1 if i > 0 else 0  # compromise from https://arxiv.org/abs/1711.07289

        for sol, J in zip(solutions, order_irreps):
            if J <= J_max:
                x = sol * window  # [m_out, m_in, x, y, z]
                basis.append(x)

    return torch.stack(basis, dim=0) if len(basis) > 0 else None


def gaussian_window(solutions, r_field, order_irreps, radii, J_max_list, sigma=.6):
    '''
    gaussian windowing function with manual handling of shell radii, shell bandlimits and shell width
    takes as input the output of _sample_cube
    :param radii: radii of the shells, sets mean of the radial gaussians
    :param J_max_list: bandlimits of the shells, same length as radii
    :param sigma: width of the shells, corresponds to standard deviation of radial gaussians
    '''
    # run over radial parts and window out non-aliased basis functions
    assert len(radii) == len(J_max_list)

    basis = []
    for r, J_max in zip(radii, J_max_list):
        window = torch.exp(-.5 * ((r_field - r) / sigma)**2) / (math.sqrt(2 * math.pi) * sigma)

        for sol, J in zip(solutions, order_irreps):
            if J <= J_max:
                x = sol * window  # [m_out, m_in, x, y, z]
                basis.append(x)

    return torch.stack(basis, dim=0) if len(basis) > 0 else None


def gaussian_window_wrapper(solutions, r_field, order_irreps, mode='compromise', border_dist=0., sigma=.6):
    '''
    convenience wrapper for windowing function with three different predefined modes for radii and bandlimits
    takes as input the output of _sample_cube
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
    size = r_field.size(0)

    n_radial = size // 2 + 1
    radii = torch.linspace(0, size // 2 - border_dist, steps=n_radial, dtype=torch.float64)

    if mode == 'conservative':
        J_max_list = [0, 2, 4, 6, 8, 10, 12, 14][:n_radial]
    if mode == 'compromise':
        J_max_list = [0, 3, 5, 7, 9, 11, 13, 15][:n_radial]
    if mode == 'sfcnn':
        J_max_list = [0, 4, 6, 8, 10, 12, 14, 16][:n_radial]

    return gaussian_window(solutions, r_field, order_irreps, radii, J_max_list, sigma)


################################################################################
# Kernel Module
################################################################################

def orthogonal_(tensor, gain=1):
    # proper orthogonal init, see https://github.com/pytorch/pytorch/pull/10672
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)

    for i in range(0, rows, cols):
        # Compute the qr factorization
        q, r = torch.qr(flattened[i:i + cols].t())
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
        q.t_()

        with torch.no_grad():
            tensor[i:i + cols].view_as(q).copy_(q)

    with torch.no_grad():
        tensor.mul_(gain)
    return tensor


class SE3Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=gaussian_window_wrapper, dyn_iso=False, verbose=False):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma
        '''
        super().__init__()

        self.size = size
        self.Rs_out = [(m, l) for m, l in Rs_out if m >= 1]
        self.Rs_in = [(m, l) for m, l in Rs_in if m >= 1]
        self.multiplicities_out = [m for m, _ in self.Rs_out]
        self.multiplicities_in = [m for m, _ in self.Rs_in]
        self.dims_out = [2 * l + 1 for _, l in self.Rs_out]
        self.dims_in = [2 * l + 1 for _, l in self.Rs_in]
        self.n_out = sum([self.multiplicities_out[i] * self.dims_out[i] for i in range(len(self.multiplicities_out))])
        self.n_in = sum([self.multiplicities_in[j] * self.dims_in[j] for j in range(len(self.multiplicities_in))])

        weights = []

        for i, (m_out, l_out) in enumerate(self.Rs_out):
            for j, (m_in, l_in) in enumerate(self.Rs_in):
                basis = cube_basis_kernels(size, l_in, l_out, radial_window)  # [beta, i, j, x, y, z]
                if basis is not None:
                    assert basis.size()[1:] == ((2 * l_out + 1), (2 * l_in + 1), size, size, size), "wrong basis shape - your cache files may probably be corrupted"

                    if verbose:
                        overlaps = torch.stack([check_basis_equivariance(basis, l_in, l_out, a, b, c) for a, b, c in torch.rand(5, 3)]).mean(0)
                        print("{} -> {} : Created {} basis elements with equivariance {}".format(l_in, l_out, len(basis), overlaps))

                    if dyn_iso:
                        # inspired by Algo. 2 in https://arxiv.org/abs/1806.05393

                        w = torch.zeros(m_out, m_in, basis.size(0))
                        if abs(l_out - l_in) == min(abs(l_out - l_in_) for _m_in, l_in_ in self.Rs_in):
                            orthogonal_(w[:, :, 0])  # only the "simplest" base operation has a non-zero init
                            # TODO this if might be called for multiple l input
                        weights += [w.flatten()]

                        basis /= basis.flatten(2).norm(dim=2).mean(1).view(basis.size(0), 1, 1, 1, 1, 1)
                    else:
                        weights += [torch.randn(m_out * m_in * basis.size(0))]

                        # rescale each basis element such that the weight can be initialized with Normal(0,1)
                        basis /= basis.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1, 1)
                        basis *= ((2 * l_out + 1) / (len(basis) * sum(self.multiplicities_in))) ** 0.5

                    self.register_buffer("kernel_{}_{}".format(i, j), basis.type(torch.get_default_dtype()))
                else:
                    self.register_buffer("kernel_{}_{}".format(i, j), None)

        self.weight = torch.nn.Parameter(torch.cat(weights))


    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, size={size})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
            size=self.size,
        )


    def combination(self, weight):
        weight_index = 0

        si_kernels = []
        begin_i = 0
        for i, mi in enumerate(self.multiplicities_out):
            i_diff = mi * self.dims_out[i]
            si = slice(begin_i, begin_i + i_diff)
            begin_j = 0
            sj_kernels = []
            for j, mj in enumerate(self.multiplicities_in):
                j_diff = mj * self.dims_in[j]
                sj = slice(begin_j, begin_j + j_diff)

                kij = getattr(self, "kernel_{}_{}".format(i, j))  # [beta, i, j, x, y, z]
                if kij is not None:
                    b_el = kij.size(0)

                    w = weight[weight_index: weight_index + mi * mj * b_el].view(mi, mj, b_el)  # [u, v, beta]
                    weight_index += mi * mj * b_el

                    ker = torch.einsum("uvb,bijxyz->uivjxyz", (w, kij)).contiguous()  # [u, i, v, j, x, y, z]
                    ker = ker.view(i_diff, j_diff, self.size, self.size, self.size)
                else:
                    ker = torch.zeros(
                        i_diff, j_diff, self.size, self.size, self.size,
                        device=weight.device, dtype=weight.dtype
                    )


                sj_kernels.append(ker)
                begin_j += mj * self.dims_in[j]

            si_kernels.append(torch.cat(sj_kernels, dim=1))
            begin_i += mi * self.dims_out[i]

        kernel = torch.cat(si_kernels, dim=0)

        assert weight_index == weight.size(0)
        return kernel

    def forward(self):  # pylint: disable=W
        return self.combination(self.weight)


################################################################################
# Measure equivariance
################################################################################

def check_basis_equivariance(basis, order_in, order_out, alpha, beta, gamma):
    from se3cnn import SO3
    from scipy.ndimage import affine_transform
    import numpy as np

    n = basis.size(0)
    dim_in = 2 * order_in + 1
    dim_out = 2 * order_out + 1
    size = basis.size(-1)
    assert basis.size() == (n, dim_out, dim_in, size, size, size), basis.size()

    basis = basis / basis.view(n, -1).norm(dim=1).view(-1, 1, 1, 1, 1, 1)

    x = basis.view(-1, size, size, size)
    y = torch.empty_like(x)

    invrot = SO3.rot(-gamma, -beta, -alpha).numpy()
    center = (np.array(x.size()[1:]) - 1) / 2

    for k in range(y.size(0)):
        y[k] = torch.tensor(affine_transform(x[k].numpy(), matrix=invrot, offset=center - np.dot(invrot, center)))

    y = y.view(*basis.size())

    y = torch.einsum(
        "ij,bjkxyz,kl->bilxyz",
        (
            irr_repr(order_out, alpha, beta, gamma, dtype=y.dtype),
            y,
            irr_repr(order_in, -gamma, -beta, -alpha, dtype=y.dtype)
        )
    )

    return torch.tensor([(basis[i] * y[i]).sum() for i in range(n)])


################################################################################
# Testing
################################################################################

def _test_basis_equivariance():
    from functools import partial
    with torch_default_dtype(torch.float64):
        basis = cube_basis_kernels(4 * 5, 2, 2, partial(gaussian_window, radii=[5], J_max_list=[999], sigma=2))
        overlaps = check_basis_equivariance(basis, 2, 2, *torch.rand(3))
        assert overlaps.gt(0.98).all(), overlaps


if __name__ == '__main__':
    _test_basis_equivariance()
