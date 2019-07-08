# pylint: disable=C,R,E1101,E1102
import torch
import se3cnn.SO3 as SO3
import math


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
def angular_function(difference_mat, l_in, l_out, filter_irreps, Ys):
    order_irreps = list(range(abs(l_in - l_out), l_in + l_out + 1))
    angular_filters = []
    for J in order_irreps:
        Y_J = get_Y_for_filter(J, filter_irreps, Ys)
        if Y_J is not None:
            # compute basis transformation matrix Q_J
            Q_J = SO3.basis_transformation_Q(J, l_in, l_out)  # [m_out * m_in, m]
            if difference_mat.dim() == 4:
                B, N, M, _ = difference_mat.size()
                K_J = torch.einsum('mn,nkab->mkab', (Q_J, Y_J))  # [m_out * m_in, batch, N, M]
                K_J = K_J.view(2 * l_out + 1, 2 * l_in + 1, B, N, M)  # [m_out, m_in, batch, N, M]
            else:
                N, M, _ = difference_mat.size()
                K_J = torch.einsum('mn,nab->mab', (Q_J, Y_J))  # [m_out * m_in, N, M]
                K_J = K_J.view(2 * l_out + 1, 2 * l_in + 1, N, M)  # [m_out, m_in, N, M]
            # Normalize wrt incoming?
            K_J = K_J.type(difference_mat.dtype)
            K_J = K_J.to(difference_mat.device)
            angular_filters.append(K_J)

    return angular_filters, difference_mat.norm(2, -1), order_irreps


# TODO: Reduce duplicate code in this and gaussian_window in kernel.py
def gaussian_radial_function(solutions, r_field, order_irreps, radii, sigma=.6, J_max=10):
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
        self.radial_function = radial_function
        self.register_buffer('radii', radii)
        self.J_filter_max = J_filter_max
        self.n_out = sum(m * (2 * l + 1) for m, l in self.Rs_out)
        self.n_in = sum(m * (2 * l + 1) for m, l in self.Rs_in)
        self.sh_backwardable = sh_backwardable

        self.nweights = 0
        set_of_irreps = set()
        filter_variances = list()
        num_paths = 0
        for m_out, l_out in self.Rs_out:
            for m_in, l_in in self.Rs_in:
                basis_size = 0
                order_irreps = list(range(abs(l_in - l_out), l_in + l_out + 1))
                for J in order_irreps:
                    if J <= self.J_filter_max:
                        basis_size += 1
                        set_of_irreps.add(J)
                # This depends on radial function
                basis_size *= len(self.radii)
                if basis_size > 0:
                    num_paths += 1
                self.nweights += m_out * m_in * basis_size
                variance_factor = (2 * l_out + 1) / (m_in * basis_size)
                filter_variances += [variance_factor ** 0.5] * (m_out * m_in * basis_size)
        self.filter_irreps = sorted(set_of_irreps)

        self.weight = torch.nn.Parameter(torch.randn(self.nweights))
        # Change variance of filter
        # We've assumed each radial function and spherical harmonic
        # is normalized to 1.
        self.register_buffer('fvar', torch.tensor(filter_variances) / num_paths ** 0.5)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, radii={radii})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
            radii=self.radii,
        )

    def combination(self, weight, difference_mat):
        # Check for batch dimension for difference_mat
        if difference_mat.dim() == 4:
            batch, N, M, _ = difference_mat.size()
            kernel = weight.new_empty(self.n_out, self.n_in, batch, N, M)
        if difference_mat.dim() == 3:
            N, M, _ = difference_mat.size()
            kernel = weight.new_empty(self.n_out, self.n_in, N, M)

        # Compute Ys for filters
        if self.sh_backwardable:
            Ys = SO3.spherical_harmonics_xyz_backwardable(self.filter_irreps, difference_mat)
        else:
            Ys = SO3.spherical_harmonics_xyz(self.filter_irreps, difference_mat)

        begin_i = 0
        weight_index = 0

        for mul_out, l_out in self.Rs_out:
            begin_j = 0
            for mul_in, l_in in self.Rs_in:
                si = slice(begin_i, begin_i + mul_out * (2 * l_out + 1))
                sj = slice(begin_j, begin_j + mul_in * (2 * l_in + 1))
                angular = angular_function(difference_mat, l_in, l_out, self.filter_irreps, Ys)  # list of [m_out, m_in, [batch], N, M]
                basis = self.radial_function(*angular, self.radii, J_max=self.J_filter_max)  # [radii * J, m_out, m_in, [batch], N, M]
                if basis is not None:
                    assert basis.size()[1:3] == (2 * l_out + 1, 2 * l_in + 1), "wrong basis shape"
                    assert basis.size()[-2:] == (N, M), "wrong basis shape"

                    b_el = basis.size(0)

                    w = weight[weight_index: weight_index + mul_out * mul_in * b_el].view(mul_out, mul_in, b_el)  # [mul_out, mul_in, radii * J]
                    weight_index += mul_out * mul_in * b_el

                    if basis.dim() == 5:
                        # uv = mul_out, mul_in
                        # r = radii * J
                        # ij = m_out m_in
                        # nm = N M
                        ker = torch.einsum("uvr,rijnm->uivjnm", (w, basis))  # [mul_out, m_out, mul_in, m_in, N, M]
                    else:
                        # k = batch
                        ker = torch.einsum("uvr,rijknm->uivjknm", (w, basis))  # [mul_out, m_out, mul_in, m_in, batch, N, M]

                    ker = ker.contiguous().view(mul_out * (2 * l_out + 1), mul_in * (2 * l_in + 1), *basis.size()[3:])

                    kernel[si, sj] = ker

                else:
                    kernel[si, sj] = 0

                begin_j += mul_in * (2 * l_in + 1)
            begin_i += mul_out * (2 * l_out + 1)
        return kernel

    def forward(self, difference_mat):  # pylint: disable=W
        return self.combination(self.weight * self.fvar, difference_mat)

