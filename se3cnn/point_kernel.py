# pylint: disable=C,R,E1101,E1102
import torch
import se3cnn.SO3 as SO3
import math
import numpy as np
import scipy.special


def get_Y_for_filter(irrep, filter_irreps, Y):
    if irrep not in filter_irreps:
        return None
    start_index = 0
    for filter_irrep in filter_irreps:
        if filter_irrep != irrep:
            start_index += 2 * filter_irrep + 1
        else:
            break
    end_index = start_index + (2 * irrep + 1)
    return Y[start_index:end_index]


#  TODO: Vectorize
def angular_function(difference_mat, order_in, order_out, filter_irreps, Ys,
                     eps=1e-8):
    order_irreps = list(range(abs(order_in - order_out),
                              order_in + order_out + 1))
    angular_filters = []
    for J in order_irreps:
        Y_J = get_Y_for_filter(J, filter_irreps, Ys)
        if Y_J is not None:
            # compute basis transformation matrix Q_J
            Q_J = SO3.basis_transformation_Q_J(J,
                                               order_in,
                                               order_out)  # [m_out * m_in, m]
            if len(difference_mat.size()) == 4:
                batch, N, M, _ = difference_mat.size()
                K_J = torch.einsum('mn,nkab->mkab',
                                   (Q_J, Y_J))  # [m_out * m_in, batch, N, M]
                K_J = K_J.reshape(2 * order_out + 1,
                                  2 * order_in + 1,
                                  batch, N, M)  # [m_out, m_in, batch, N, M]
            else:
                N, M, _ = difference_mat.size()
                K_J = torch.einsum('mn,nab->mab',
                                   (Q_J, Y_J))  # [m_out * m_in, N, M]
                K_J = K_J.reshape(2 * order_out + 1,
                                  2 * order_in + 1,
                                  N, M)  # [m_out, m_in, N, M]
            # Normalize wrt incoming?
            angular_filters.append(K_J)

    return angular_filters, difference_mat.norm(2, -1), order_irreps


# TODO: Reduce duplicate code in this and gaussian_window in kernel.py
def gaussian_radial_function(solutions, r_field, order_irreps, radii, sigma=.6,
                             J_max=10):
    '''
    gaussian radial function with  manual handling of shell radii, shell
    bandlimits and shell width takes as input the output of angular_function
    :param radii: radii of the shells, sets mean of the radial gaussians
    :param sigma: width of the shells, corresponds to standard deviation of
        radial gaussians
    '''
    basis = []
    for r in radii:
        window = torch.exp(-.5 * ((r_field - r) / sigma)**2)
        window = window / (math.sqrt(2 * math.pi) * sigma)

        for sol, J in zip(solutions, order_irreps):
            if J <= J_max:
                x = sol.to(window.device) * window  # [m_out, m_in, x, y, z]
                basis.append(x)

    return torch.stack(basis, dim=0) if len(basis) > 0 else None


def gaussian_radial_function_normed(solutions, r_field, order_irreps, radii,
                                    sigma=.6, J_max=10):
    '''
    gaussian radial function with  manual handling of shell radii, shell
    bandlimits and shell width takes as input the output of angular_function
    :param radii: radii of the shells, sets mean of the radial gaussians
    :param sigma: width of the shells, corresponds to standard deviation of
        radial gaussians
    '''
    basis = []
    for r in radii:
        window = torch.exp(-.5 * ((r_field - r) / sigma)**2)
        window /= (2 * r ** 2 + 1)
        window = window / (math.sqrt(2 * math.pi) * sigma)

        for sol, J in zip(solutions, order_irreps):
            if J <= J_max:
                x = sol.to(window.device) * window  # [m_out, m_in, x, y, z]
                basis.append(x)

    return torch.stack(basis, dim=0) if len(basis) > 0 else None


# TODO: Split into radial and angular kernels
class SE3PointKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, radii,
                 radial_function=gaussian_radial_function, J_filter_max=10,
                 sh_backwardable=False):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation
        alpha, beta, gamma
        '''
        super().__init__()

        self.Rs_out = [(m, l) for m, l in Rs_out if m >= 1]
        self.Rs_in = [(m, l) for m, l in Rs_in if m >= 1]
        self.multiplicities_out = [m for m, _ in self.Rs_out]
        self.multiplicities_in = [m for m, _ in self.Rs_in]
        self.dims_out = [2 * l + 1 for _, l in self.Rs_out]
        self.dims_in = [2 * l + 1 for _, l in self.Rs_in]
        self.radial_function = radial_function
        self.register_buffer('radii', radii)
        self.J_filter_max = J_filter_max
        self.n_out = sum([self.multiplicities_out[i] * self.dims_out[i] for i
                          in range(len(self.multiplicities_out))])
        self.n_in = sum([self.multiplicities_in[j] * self.dims_in[j] for j in
                         range(len(self.multiplicities_in))])
        self.sh_backwardable = sh_backwardable

        self.nweights = 0
        set_of_irreps = set()
        filter_variances = list()
        num_paths = 0
        for i, (m_out, l_out) in enumerate(self.Rs_out):
            for j, (m_in, l_in) in enumerate(self.Rs_in):
                basis_size = 0
                for _ in self.radii:
                    order_irreps = list(range(abs(l_in - l_out),
                                              l_in + l_out + 1))
                    for J in order_irreps:
                        if J <= self.J_filter_max:
                            basis_size += 1
                            set_of_irreps.add(J)
                # This depends on radial function
                if basis_size > 0:
                    num_paths += 1
                self.nweights += m_out * m_in * basis_size
                variance_factor = (2 * l_out + 1) / (m_in * basis_size)
                filter_variances += [np.sqrt(variance_factor)] * (m_out *
                                                                  m_in *
                                                                  basis_size)
        self.filter_irreps = sorted(list(set_of_irreps))

        self.weight = torch.nn.Parameter(torch.randn(self.nweights))
        # Change variance of filter
        # We've assumed each radial function and spherical harmonic
        # is normalized to 1.
        self.register_buffer('fvar', (torch.tensor(filter_variances) *
                                      np.sqrt(1 / num_paths)))

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, radii={radii})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
            radii=self.radii,
        )

    def combination(self, weight, difference_mat):
        # Check for batch dimension for difference_mat
        if len(difference_mat.size()) == 4:
            batch, N, M, _ = difference_mat.size()
            kernel = weight.new_empty(self.n_out, self.n_in, batch, N, M)
        if len(difference_mat.size()) == 3:
            N, M, _ = difference_mat.size()
            kernel = weight.new_empty(self.n_out, self.n_in, N, M)

        begin_i = 0
        weight_index = 0

        # Compute Ys for filters
        if self.sh_backwardable:
            Ys = SO3.spherical_harmonics_xyz_backwardable_order_list(
                self.filter_irreps, difference_mat)
        else:
            Ys = SO3.spherical_harmonics_xyz(
                self.filter_irreps, difference_mat)

        for i, (m_out, l_out) in enumerate(self.Rs_out):
            begin_j = 0
            for j, (m_in, l_in) in enumerate(self.Rs_in):
                si = slice(begin_i, begin_i + m_out * self.dims_out[i])
                sj = slice(begin_j, begin_j + m_in * self.dims_in[j])
                angular = angular_function(difference_mat, l_in, l_out,
                                           self.filter_irreps, Ys)
                basis = self.radial_function(*angular, self.radii,
                                             J_max=self.J_filter_max)
                if basis is not None:
                    assert basis.size()[1:3] == ((2 * l_out + 1), (2 * l_in + 1)), "wrong basis shape"
                    assert basis.size()[-2:] == (N, M), "wrong basis shape"
                    kij = basis

                    b_el = kij.size(0)
                    b_size = kij.size()[1:]  # [i, j, N, M] or [i, j, batch, N, M]

                    w = weight[weight_index: weight_index + m_out * m_in * b_el].view(m_out * m_in, b_el)  # [I*J, beta]
                    weight_index += m_out * m_in * b_el

                    basis_kernels_ij = kij.contiguous().view(b_el, -1)  # [beta, i*j*N*M] or [beta, i*j*batch*N*M]

                    # TODO: Rewrite as einsum
                    ker = torch.mm(w, basis_kernels_ij)  # [I*J, i*j*N*M] or [I*J, i*j*batch*N*M]
                    ker = ker.view(m_out, m_in, *b_size)  # [I, J, i, j, N, M] or [I, J, i, j, batch, N, M]
                    ker = ker.transpose(1, 2).contiguous()  # [I, i, J, j, N, M] or [I, i, J, j, batch, N, M]
                    ker = ker.view(m_out * self.dims_out[i], m_in * self.dims_in[j], *b_size[2:])  # [I*i, J*j, N, M] or [I*i, J*j, batch, N, M]
                    kernel[si, sj] = ker

                else:
                    kernel[si, sj] = 0

                begin_j += m_in * self.dims_in[j]
            begin_i += m_out * self.dims_out[i]
        return kernel

    def forward(self, difference_mat):  # pylint: disable=W
        return self.combination(self.weight * self.fvar, difference_mat)

