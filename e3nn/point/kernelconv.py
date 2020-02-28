import math
import torch

import e3nn.o3 as o3
import e3nn.rs as rs


class KernelConv(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, get_l_filters=o3.selection_rule, sh=o3.spherical_harmonics_xyz,
                 normalization='norm'):
        '''
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param get_l_filters: function of signature (l_in, l_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        TODO fix doc
        '''
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        def filters_with_parity(l_in, p_in, l_out, p_out):
            nonlocal get_l_filters
            return [l for l in get_l_filters(l_in, l_out) if p_out == 0 or p_in * (-1) ** l == p_out]

        self.get_l_filters = filters_with_parity
        self.check_input_output()
        self.sh = sh

        assert isinstance(normalization, str), "normalization should be passed as a string value"
        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

        def lm_normalization(l_out, l_in):
            # put 2l_in+1 to keep the norm of the m vector constant
            # put 2l_ou+1 to keep the variance of each m component constant
            # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
            lm_norm = None
            if normalization == 'norm':
                lm_norm = math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)
            elif normalization == 'component':
                lm_norm = math.sqrt(2 * l_out + 1) * math.sqrt(4 * math.pi)
            return lm_norm

        norm_coef = torch.zeros((len(self.Rs_out), len(self.Rs_in), 2))

        n_path = 0
        set_of_l_filters = set()

        for i, (mul_out, l_out, p_out) in enumerate(self.Rs_out):
            # consider that we sum a bunch of [lambda_(m_out)] vectors
            # we need to count how many of them we sum in order to normalize the network
            num_summed_elements = 0
            for mul_in, l_in, p_in in self.Rs_in:
                l_filters = self.get_l_filters(l_in, p_in, l_out, p_out)
                num_summed_elements += mul_in * len(l_filters)

            for j, (mul_in, l_in, p_in) in enumerate(self.Rs_in):
                # normalization assuming that each terms are of order 1 and uncorrelated
                norm_coef[i, j, 0] = lm_normalization(l_out, l_in) / math.sqrt(num_summed_elements)
                norm_coef[i, j, 1] = lm_normalization(l_out, l_in) / math.sqrt(mul_in)

                l_filters = self.get_l_filters(l_in, p_in, l_out, p_out)
                assert l_filters == sorted(set(l_filters)), "get_l_filters must return a sorted list of unique values"

                # compute the number of degrees of freedom
                n_path += mul_out * mul_in * len(l_filters)

                # create the set of all spherical harmonics orders needed
                set_of_l_filters = set_of_l_filters.union(l_filters)

        # create the radial model: R+ -> R^n_path
        # it contains the learned parameters
        self.R = RadialModel(n_path)
        self.set_of_l_filters = sorted(set_of_l_filters)
        self.register_buffer('norm_coef', norm_coef)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_input_output(self):
        for _, l_out, p_out in self.Rs_out:
            has_path = False
            for _, l_in, p_in in self.Rs_in:
                if self.get_l_filters(l_in, p_in, l_out, p_out):
                    has_path = True
                    break
            if not has_path:
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

        for _, l_in, p_in in self.Rs_in:
            has_path = False
            for _, l_out, p_out in self.Rs_out:
                if self.get_l_filters(l_in, p_in, l_out, p_out):
                    has_path = True
                    break
            if not has_path:
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def forward(self, features, geometry):
        """
        :param features: tensor [batch, in_point, l_in * mul_in * m_in] TODO fix doc
        :param geometry: tensor [batch, in_point, xyz]
        :return:         tensor [batch, out_point, l_out * mul_out * m_out]
        """
        *size, n_atoms, xyz = geometry.size()
        assert xyz == 3
        assert geometry.ndim == 3

        # TODO consider putting this outside somehow?
        rb = geometry.unsqueeze(-2)  # [..., 1, n_atom, xyz]
        ra = geometry.unsqueeze(-3)  # [..., a, 1, xyz]
        r = (rb - ra).reshape(-1, n_atoms, xyz)  # [... * a, b, xyz]
        # Henceforth [... * a] is known as batch!!

        # precompute all needed spherical harmonics
        Y = self.sh(self.set_of_l_filters, r)  # [l_filter * m_filter, batch, n_atom]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        radii = r.norm(2, dim=-1)  # [batch, n_atoms]
        R = self.R(radii.flatten()).reshape(*radii.shape, -1)  # [batch, n_atoms, l_out * l_in * mul_out * mul_in *  l_filter]

        norm_coef = getattr(self, 'norm_coef')
        norm_coef = norm_coef[:, :, (radii == 0).type(torch.long)]  # [l_out, l_in, batch, n_atom]

        # kernel_conv = KernelFn.apply(
        #     features, Y, R, norm_coef, self.Rs_in, self.Rs_out, self.get_l_filters, self.set_of_l_filters
        # )
        kernel_conv = kernel_conv_automatic_backward(
            features, Y, R, norm_coef, self.Rs_in, self.Rs_out, self.get_l_filters, self.set_of_l_filters
        )
        return kernel_conv.view(*size, n_atoms, kernel_conv.shape[1])


def kernel_conv_automatic_backward(features, Y, R, norm_coef, Rs_in, Rs_out, get_l_filters, set_of_l_filters):
    n_in = rs.dim(Rs_in)
    n_out = rs.dim(Rs_out)
    batch, n_atom = Y.shape[1:]

    features = features.reshape(batch, n_in)
    kernel_conv = Y.new_zeros(batch, n_out)

    # note: for the normalization we assume that the variance of R[i] is one
    begin_R = 0

    begin_out = 0
    for i, (mul_out, l_out, p_out) in enumerate(Rs_out):
        len_s_out = mul_out * (2 * l_out + 1)
        s_out = slice(begin_out, begin_out + len_s_out)
        begin_out += mul_out * (2 * l_out + 1)

        begin_in = 0
        for j, (mul_in, l_in, p_in) in enumerate(Rs_in):
            len_s_in = mul_in * (2 * l_in + 1)
            s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
            begin_in += mul_in * (2 * l_in + 1)

            l_filters = get_l_filters(l_in, p_in, l_out, p_out)
            if not l_filters:
                continue

            # extract the subset of the `R` that corresponds to the couple (l_out, l_in)
            n = mul_out * mul_in * len(l_filters)
            sub_R = R[..., begin_R: begin_R + n].view(batch, n_atom, mul_out, mul_in, -1)  # [batch, n_atom, mul_out, mul_in, l_filter]
            begin_R += n

            sub_norm_coef = norm_coef[i, j, :]  # [batch, n_atom]

            K = 0
            for k, l_filter in enumerate(l_filters):
                tmp = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
                sub_Y = Y[tmp: tmp + 2 * l_filter + 1, :]  # [m, batch, n_atom]

                C = o3.clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel_conv)  # [m_out, m_in, m]

                K += torch.einsum(
                    "ijk,kza,zauv,za,zvj->zui",
                    C, sub_Y, sub_R[..., k], sub_norm_coef, features[..., s_in].reshape(batch, mul_in, -1)
                )  # [batch, mul_out, m_out]

            if K is not 0:
                kernel_conv[:, s_out] += K.reshape(batch, -1)
            else:
                raise NotImplementedError("Does this even happen?")
    return kernel_conv


class KernelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, Y, R, norm_coef, Rs_in, Rs_out, get_l_filters, set_of_l_filters):
        """
        :param features: tensor [batch, a, l_in * mul_in * m_in]
        :param Y: tensor [l_filter * m_filter, batch]  TODO fix these
        :param R: tensor [batch, l_out * l_in * mul_out * mul_in * l_filter]
        :param norm_coef: tensor [l_out, l_in, batch]
        :return: tensor [batch, l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        # FUNCTION ABOVE IS UPDATED

        # ctx.Rs_in = Rs_in
        # ctx.Rs_out = Rs_out
        # ctx.get_l_filters = get_l_filters
        # ctx.set_of_l_filters = set_of_l_filters
        #
        # # save necessary tensors for backward
        # saved_Y = saved_R = None
        # if Y.requires_grad:
        #     ctx.Y_shape = Y.shape
        #     saved_R = R
        # if R.requires_grad:
        #     ctx.R_shape = R.shape
        #     saved_Y = Y
        # ctx.save_for_backward(saved_Y, saved_R, norm_coef)
        #
        # batch, n_a, n_b = Y.shape[1:]
        # n_in = rs.dim(ctx.Rs_in)
        # n_out = rs.dim(ctx.Rs_out)
        #
        # kernel_conv = Y.new_zeros(batch, n_a, n_out)
        #
        # print(kernel_conv.shape)
        # print(ctx.Rs_in)
        # print(ctx.Rs_out)
        # for b in range(n_b):
        #     # note: for the normalization we assume that the variance of R[i] is one
        #     begin_R = 0
        #
        #     begin_out = 0
        #     for i, (mul_out, l_out, p_out) in enumerate(ctx.Rs_out):
        #         len_s_out = mul_out * (2 * l_out + 1)
        #         s_out = slice(begin_out, begin_out + len_s_out)
        #         begin_out += mul_out * (2 * l_out + 1)
        #
        #         begin_in = 0
        #         for j, (mul_in, l_in, p_in) in enumerate(ctx.Rs_in):
        #             len_s_in = mul_in * (2 * l_in + 1)
        #             s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
        #             begin_in += mul_in * (2 * l_in + 1)
        #
        #             l_filters = ctx.get_l_filters(l_in, p_in, l_out, p_out)
        #             if not l_filters:
        #                 continue
        #
        #             # extract the subset of the `R` that corresponds to the couple (l_out, l_in)
        #             n = mul_out * mul_in * len(l_filters)
        #             sub_R = R[..., b, begin_R: begin_R + n].view(batch, n_a, mul_out, mul_in, -1)  # [batch, mul_out, mul_in, l_filter]
        #             begin_R += n
        #
        #             # TODO figure out norm
        #             # sub_norm_coef = norm_coef[i, j]  # [batch]
        #
        #             K = 0
        #             for k, l_filter in enumerate(l_filters):
        #                 tmp = sum(2 * l + 1 for l in ctx.set_of_l_filters if l < l_filter)
        #                 sub_Y = Y[tmp: tmp + 2 * l_filter + 1, :, b]  # [m, batch, a]
        #
        #                 C = o3.clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel_conv)  # [m_out, m_in, m]
        #
        #                 K += torch.einsum(
        #                     "ijk,kza,zauv->zauivj", C, sub_Y, sub_R[..., k]
        #                 )  # [batch, a, mul_out, m_out, mul_in, m_in]
        #
        #             if K is not 0:
        #                 kernel_conv[:, :, s_out] += torch.einsum(
        #                     'zaj,zaij->zai',
        #                     features[:, :, s_in],
        #                     K.reshape(batch, n_a, len_s_out, len_s_in)
        #                 )
        #             else:
        #                 raise NotImplementedError("Does this even happen?")
        #
        # return kernel_conv

    @staticmethod
    def backward(ctx, grad_kernel):
        raise NotImplementedError("Not yet!")
