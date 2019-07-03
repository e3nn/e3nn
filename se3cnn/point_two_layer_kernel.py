# pylint: disable=C,R,E1101,E1102
import torch
from se3cnn.SO3 import x_to_alpha_beta, irr_repr, spherical_harmonics, kron, spherical_harmonics_xyz, basis_transformation_Q_J
import se3cnn.SO3 as SO3
from se3cnn.point_kernel import get_Y_for_filter, angular_function, gaussian_radial_function
import math
import torch.nn.functional as F
import numpy as np


# TODO: Split into radial and angular kernels
class SE3PointTwoLayerKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, radii,
                 radial_function=gaussian_radial_function, J_filter_max=10,
                 radial_nonlinearity=None, hidden_dim=10, sh_backwardable=False):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma
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
        self.hidden_dim = hidden_dim

        self.radial_nonlinearity = F.relu if radial_nonlinearity is None else radial_nonlinearity

        self.nweights_0 = 0
        self.nweights_1 = 0
        self.nbiases_0 = 0
        self.nbiases_1 = 0

        set_of_irreps = set()
        filter_variances_0 = list()
        filter_variances_1 = list()
        num_paths = 0
        for i, (m_out, l_out) in enumerate(self.Rs_out):
            for j, (m_in, l_in) in enumerate(self.Rs_in):
                basis_size = 0
                for _ in self.radii:
                    order_irreps = list(range(abs(l_in - l_out), l_in + l_out + 1))
                    for J in order_irreps:
                        if J <= self.J_filter_max:
                            basis_size += 1
                            set_of_irreps.add(J)
                if basis_size > 0:
                    num_paths += 1
                self.nweights_0 += self.hidden_dim * basis_size  # This depends on radial function
                variance_factor_0 = (2 * l_out + 1) / (self.hidden_dim * basis_size)
                filter_variances_0 += [np.sqrt(variance_factor_0)] * (self.hidden_dim * basis_size)
                self.nbiases_0 += self.hidden_dim

                self.nweights_1 += m_out * m_in * self.hidden_dim
                variance_factor_1 = (2 * l_out + 1) / (m_in * self.hidden_dim)
                filter_variances_1 += [np.sqrt(variance_factor_1)] * (m_out *
                                                                      m_in *
                                                                      self.hidden_dim)
                self.nbiases_1 += m_out * m_in
        self.filter_irreps = sorted(list(set_of_irreps))

        self.weight_0 = torch.nn.Parameter(torch.randn(self.nweights_0))
        self.biases_0 = torch.nn.Parameter(torch.zeros(self.nbiases_0))
        self.weight_1 = torch.nn.Parameter(torch.randn(self.nweights_1))
        self.biases_1 = torch.nn.Parameter(torch.zeros(self.nbiases_1))

        self.register_buffer('fvar0', (torch.tensor(filter_variances_0) *
                                       np.sqrt(1 / num_paths)))
        self.register_buffer('fvar1', (torch.tensor(filter_variances_1) *
                                       np.sqrt(1 / num_paths)))

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, radii={radii})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
            radii=self.radii,
        )


    def combination(self, weight_0, weight_1, biases_0, biases_1, difference_mat):
        # Check for batch dimension for difference_mat
        if len(difference_mat.size()) == 4:
            batch, N, M, _ = difference_mat.size()
            kernel = weight_0.new_empty(self.n_out, self.n_in, batch, N, M)
        if len(difference_mat.size()) == 3:
            N, M, _ = difference_mat.size()
            kernel = weight_0.new_empty(self.n_out, self.n_in, N, M)

        begin_i = 0
        weight_0_index = 0
        weight_1_index = 0
        bias_0_index = 0
        bias_1_index = 0

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
                angular = angular_function(difference_mat, l_in, l_out, self.filter_irreps, Ys)
                basis = self.radial_function(*angular, self.radii, J_max=self.J_filter_max)
                if basis is not None:
                    assert basis.size()[1:3] == ((2 * l_out + 1), (2 * l_in + 1)), "wrong basis shape"
                    assert basis.size()[-2:] == (N, M), "wrong basis shape"
                    kij = basis

                    b_el = kij.size(0)
                    b_size = kij.size()[1:]  # [i, j, N, M] or [i, j, batch, N, M]

                    w0 = weight_0[weight_0_index: weight_0_index + self.hidden_dim * b_el].view(self.hidden_dim, b_el)  # [I*J, beta]
                    b0 = biases_0[bias_0_index: bias_0_index + self.hidden_dim]  # [I*J]
                    w1 = weight_1[weight_1_index: weight_1_index + m_out * m_in * self.hidden_dim].view(m_out * m_in, self.hidden_dim)  # [I*J, I*J]
                    b1 = biases_1[bias_1_index: bias_1_index + m_out * m_in]  # [I*J]

                    # weight_index += m_out * m_in * b_el
                    weight_0_index += self.hidden_dim * b_el
                    bias_0_index += self.hidden_dim
                    weight_1_index += m_out * m_in * self.hidden_dim
                    bias_1_index += m_out * m_in

                    basis_kernels_ij = kij.contiguous().view(b_el, -1)  # [beta, i*j*N*M] or [beta, i*j*batch*N*M]

                    ker = torch.mm(w0, basis_kernels_ij)  # [hidden_dim, i*j*N*M] or [hidden_dim, i*j*batch*N*M]
                    # Add bias and nonlinearity
                    ker += b0.unsqueeze(-1)
                    ker = self.radial_nonlinearity(ker)
                    # Add weight and bias
                    ker = torch.mm(w1, ker)  # [I*J, i*j*N*M] or [I*J, i*j*batch*N*M]
                    ker += b1.unsqueeze(-1)

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
        return self.combination(self.weight_0 * self.fvar0, self.weight_1 *
                                self.fvar1, self.biases_0, self.biases_1, difference_mat)
