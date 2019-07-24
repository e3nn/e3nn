# pylint: disable=C, R, arguments-differ, no-member
import math

import torch

import se3cnn.SO3 as SO3


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, get_l_filters=None, sh=None, normalization='norm'):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param get_l_filters: function of signature (l_in, l_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        '''
        super().__init__()

        self.Rs_out = [(mul, l) for mul, l in Rs_out if mul >= 1]
        self.Rs_in = [(mul, l) for mul, l in Rs_in if mul >= 1]
        self.n_out = sum(mul * (2 * l + 1) for mul, l in self.Rs_out)
        self.n_in = sum(mul * (2 * l + 1) for mul, l in self.Rs_in)

        if get_l_filters is None:
            get_l_filters = lambda l_in, l_out: list(range(abs(l_in - l_out), l_in + l_out + 1))
        self.get_l_filters = get_l_filters

        if sh is None:
            sh = SO3.spherical_harmonics_xyz
        self.sh = sh

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

        n_path = 0
        set_of_l_filters = set()

        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                l_filters = self.get_l_filters(l_in, l_out)
                assert l_filters == sorted(set(l_filters)), "get_l_filters must return a sorted list of unique values"

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

    def forward(self, r):
        """
        :param r: tensor [batch, 3]
        :return: tensor [batch, l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        batch, xyz = r.size()
        assert xyz == 3

        kernel = r.new_zeros(batch, self.n_out, self.n_in)

        # precompute all needed spherical harmonics
        Ys = self.sh(self.set_of_l_filters, r)  # [l_filter * m_filter, batch]

        # use the radial model to fix all the degrees of freedom
        radii = r.norm(2, dim=1)  # [batch]
        # note: for the normalization we assume that the variance of coefficients[i] is one
        coefficients = self.R(radii)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]
        begin_c = 0

        begin_out = 0
        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))

            # consider that we sum a bunch of [lambda_(m_out)] vectors
            # we need to count how many of them we sum in order to normalize the network
            num_summed_elements = 0
            for mul_in, l_in in self.Rs_in:
                l_filters = self.get_l_filters(l_in, l_out)
                num_summed_elements += mul_in * len(l_filters)

            begin_in = 0
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))

                l_filters = self.get_l_filters(l_in, l_out)

                # extract the subset of the `coefficients` that corresponds to the couple (l_out, l_in)
                n = mul_out * mul_in * len(l_filters)
                c = coefficients[:, begin_c: begin_c + n].contiguous().view(batch, mul_out, mul_in, -1)  # [batch, mul_out, mul_in, l_filter]
                begin_c += n

                Qs = getattr(self, "Q_{}_{}".format(i, j))  # [m_out, m_in, l_filter * m_filter]

                # note: I don't know if we can vectorize this for loop because [l_filter * m_filter] cannot be put into [l_filter, m_filter]
                K = 0
                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in self.set_of_l_filters if l < l_filter)
                    Y = Ys[tmp: tmp + 2 * l_filter + 1]  # [m, batch]

                    tmp = sum(2 * l + 1 for l in l_filters if l < l_filter)
                    Q = Qs[:, :, tmp: tmp + 2 * l_filter + 1]  # [m_out, m_in, m]

                    # note: The multiplication with `c` could also be done outside of the for loop
                    K += torch.einsum("ijk,kz,zuv->zuivj", (Q, Y, c[..., k]))  # [batch, mul_out, m_out, mul_in, m_in]

                # put 2l_in+1 to keep the norm of the m vector constant
                # put 2l_ou+1 to keep the variance of each m componant constant
                # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
                if self.normalization == 'norm':
                    x = math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)
                if self.normalization == 'component':
                    x = math.sqrt(2 * l_out + 1) * math.sqrt(4 * math.pi)

                # normalization assuming that each terms are of order 1 and uncorrelated
                x /= num_summed_elements ** 0.5

                # TODO create tests for these normalizations
                K.mul_(x)

                if K is not 0:
                    kernel[:, s_out, s_in] = K.contiguous().view_as(kernel[:, s_out, s_in])

                begin_in += mul_in * (2 * l_in + 1)
            begin_out += mul_out * (2 * l_out + 1)

        return kernel
