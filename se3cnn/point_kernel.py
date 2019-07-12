# pylint: disable=C, R, arguments-differ, no-member
import math

import torch

import se3cnn.SO3 as SO3


class SE3PointKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, l_filter_max=10, sh_backwardable=False):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        :param RadialModel: Class(d), trainable model: R -> R^d
        '''
        super().__init__()

        self.Rs_out = [(mul, l) for mul, l in Rs_out if mul >= 1]
        self.Rs_in = [(mul, l) for mul, l in Rs_in if mul >= 1]
        self.n_out = sum(mul * (2 * l + 1) for mul, l in self.Rs_out)
        self.n_in = sum(mul * (2 * l + 1) for mul, l in self.Rs_in)

        self.l_filter_max = l_filter_max
        self.sh_backwardable = sh_backwardable

        n_path = 0
        set_of_l_filters = set()

        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                l_filters = self.get_l_filters(l_in, l_out)

                # compute the number of degrees of freedom
                n_path += mul_out * mul_in * len(l_filters)

                # create the set of all spherical harmonics orders needed
                set_of_l_filters = set_of_l_filters.union(l_filters)

                # precompute the change of basis Q
                Q = [SO3.clebsch_gordan(l_out, l_in, l) for l in l_filters]
                Q = torch.cat(Q, dim=2)  # [m_out, m_in, l_filter * m_filter]
                self.register_buffer("Q_{}_{}".format(i, j), Q.type(torch.get_default_dtype()))

        # create the radial model: R+ -> R^n_path
        # it contains the learned parameters
        self.R = RadialModel(n_path)

        self.set_of_l_filters = sorted(set_of_l_filters)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
        )

    def get_l_filters(self, l_in, l_out):
        ls = list(range(abs(l_in - l_out), l_in + l_out + 1))
        ls = [l for l in ls if l <= self.l_filter_max]
        return ls

    def forward(self, difference_matrix):
        """
        :param difference_matrix: tensor [[batch,] N_out, N_in, 3]
        :return: tensor [l_out * mul_out * m_out, l_in * mul_in * m_in, [batch,] N_out, N_in]
        """
        has_batch = difference_matrix.dim() == 4
        if not has_batch:
            difference_matrix = difference_matrix.unsqueeze(0)

        batch, N_out, N_in, _ = difference_matrix.size()

        kernel = difference_matrix.new_zeros(self.n_out, self.n_in, batch, N_out, N_in)

        # precompute all needed spherical harmonics
        sh = SO3.spherical_harmonics_xyz_backwardable if self.sh_backwardable else SO3.spherical_harmonics_xyz
        Ys = sh(self.set_of_l_filters, difference_matrix)  # [l_filter * m_filter, batch, N_out, N_in]

        # use the radial model to fix all the degrees of freedom
        radii = difference_matrix.norm(2, dim=-1).view(-1)  # [batch * N_out * N_in]
        # note: for the normalization we assume that the variance of weights[i] is one
        weights = self.R(radii).view(batch, N_out, N_in, -1)  # [batch, N_out, N_in, l_out * l_in * mul_out * mul_in * l_filter]
        begin_w = 0

        begin_out = 0
        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))

            # consider that we sum a bunch of [lambda_(m_out)] vectors
            # we need to count how many of them we sum in order to normalize the network
            num_summed_elements = 0
            for mul_in, l_in in self.Rs_in:
                l_filters = self.get_l_filters(l_in, l_out)
                num_summed_elements += mul_in * len(l_filters)
            num_summed_elements *= N_in  # note: idealy the number of neighbours

            begin_in = 0
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))

                l_filters = self.get_l_filters(l_in, l_out)

                # extract the subset of the `weights` that corresponds to the couple (l_out, l_in)
                n = mul_out * mul_in * len(l_filters)
                w = weights[:, :, :, begin_w: begin_w + n].contiguous().view(batch, N_out, N_in, mul_out, mul_in, -1)  # [batch, N_out, N_in, mul_out, mul_in, l_filter]
                begin_w += n

                Qs = getattr(self, "Q_{}_{}".format(i, j))  # [m_out, m_in, l_filter * m_filter]

                # note: I don't know if we can vectorize this for loop because [l_filter * m_filter] cannot be put into [l_filter, m_filter]
                K = 0
                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in self.set_of_l_filters if l < l_filter)
                    Y = Ys[tmp: tmp + 2 * l_filter + 1]  # [m, batch, N_out, N_in]

                    tmp = sum(2 * l + 1 for l in l_filters if l < l_filter)
                    Q = Qs[:, :, tmp: tmp + 2 * l_filter + 1]  # [m_out, m_in, m]

                    # note: The multiplication with `w` could also be done outside of the for loop
                    K += torch.einsum("ijr,rknm,knmuv->uivjknm", (Q, Y, w[..., k]))  # [mul_out, m_out, mul_in, m_in, batch, N_out, N_in]

                # put 2l_in+1 to keep the norm of the m vector constant
                # put 2l_ou+1 to keep the variance of each m componant constant
                # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
                K *= math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)

                # normalization assuming that each terms are of order 1 and uncorrelated
                K /= num_summed_elements ** 0.5

                if K is not 0:
                    kernel[s_out, s_in] = K.contiguous().view_as(kernel[s_out, s_in])

                begin_in += mul_in * (2 * l_in + 1)
            begin_out += mul_out * (2 * l_out + 1)

        if not has_batch:
            kernel = kernel.squeeze(2)

        return kernel
