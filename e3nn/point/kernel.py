# pylint: disable=C, R, arguments-differ, no-member
import math
import torch

import e3nn.SO3 as SO3


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, get_l_filters=None, sh=SO3.spherical_harmonics_xyz, normalization='norm'):
        '''
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param get_l_filters: function of signature (l_in, l_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        '''
        super().__init__()

        self.Rs_in = SO3.normalizeRs(Rs_in)
        self.Rs_out = SO3.normalizeRs(Rs_out)

        def filters_with_parity(l_in, p_in, l_out, p_out):
            def filters(l_in, l_out):
                return list(range(abs(l_in - l_out), l_in + l_out + 1))

            nonlocal get_l_filters
            get_l_filters = filters if get_l_filters is None else get_l_filters
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
            Rs_in=SO3.formatRs(self.Rs_in),
            Rs_out=SO3.formatRs(self.Rs_out),
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

    def forward(self, r):
        """
        :param r: tensor [..., 3]
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        *size, xyz = r.size()
        assert xyz == 3
        r = r.view(-1, 3)

        # use the radial model to fix all the degrees of freedom
        radii = r.norm(2, dim=1)  # [batch]

        # precompute all needed spherical harmonics
        Ys = self.sh(self.set_of_l_filters, r)  # [l_filter * m_filter, batch]

        # note: for the normalization we assume that the variance of coefficients[i] is one
        coefficients = self.R(radii)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]
        norm_coef = getattr(self, 'norm_coef')
        norm_coef = norm_coef[:, :, (radii == 0).type(torch.long)]  # [l_out, l_in, batch]

        kernel = KernelFn.apply(Ys, coefficients, norm_coef, self.Rs_in, self.Rs_out, self.get_l_filters, self.set_of_l_filters)
        return kernel.view(*size, kernel.shape[1], kernel.shape[2])


class KernelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ys, coefficients, norm_coef, Rs_in, Rs_out, get_l_filters, set_of_l_filters):
        """
        :param Ys: tensor [l_filter * m_filter, batch]
        :param coefficients: tensor [batch, l_out * l_in * mul_out * mul_in * l_filter]
        :param norm_coef: tensor [l_out, l_in, batch]
        :return: tensor [batch, l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        ctx.Rs_in = Rs_in
        ctx.Rs_out = Rs_out
        ctx.get_l_filters = get_l_filters
        ctx.set_of_l_filters = set_of_l_filters
        ctx.Ys_shape = Ys.shape
        ctx.coefficients_shape = coefficients.shape

        saved_Ys = saved_coefficients = None
        if Ys.requires_grad:
            saved_coefficients = coefficients
        if coefficients.requires_grad:
            saved_Ys = Ys

        ctx.save_for_backward(saved_Ys, saved_coefficients, norm_coef)

        batch = Ys.shape[1]
        n_in = sum(mul * (2 * l + 1) for mul, l, _ in ctx.Rs_in)
        n_out = sum(mul * (2 * l + 1) for mul, l, _ in ctx.Rs_out)

        kernel = Ys.new_zeros(batch, n_out, n_in)

        # note: for the normalization we assume that the variance of coefficients[i] is one
        begin_c = 0

        begin_out = 0
        for i, (mul_out, l_out, p_out) in enumerate(ctx.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
            begin_out += mul_out * (2 * l_out + 1)

            begin_in = 0
            for j, (mul_in, l_in, p_in) in enumerate(ctx.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
                begin_in += mul_in * (2 * l_in + 1)

                l_filters = ctx.get_l_filters(l_in, p_in, l_out, p_out)
                if not l_filters:
                    continue

                # extract the subset of the `coefficients` that corresponds to the couple (l_out, l_in)
                n = mul_out * mul_in * len(l_filters)
                c = coefficients[:, begin_c: begin_c + n].contiguous().view(batch, mul_out, mul_in, -1)  # [batch, mul_out, mul_in, l_filter]
                begin_c += n

                n = norm_coef[i, j]  # [batch]

                # note: I don't know if we can vectorize this for loop because [l_filter * m_filter] cannot be put into [l_filter, m_filter]
                K = 0
                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in ctx.set_of_l_filters if l < l_filter)
                    Y = Ys[tmp: tmp + 2 * l_filter + 1]  # [m, batch]

                    C = SO3.clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel)  # [m_out, m_in, m]

                    # note: The multiplication with `c` could also be done outside of the for loop
                    K += torch.einsum("ijk,kz,zuv,z->zuivj", (C, Y, c[..., k], n))  # [batch, mul_out, m_out, mul_in, m_in]

                if K is not 0:
                    kernel[:, s_out, s_in] = K.contiguous().view_as(kernel[:, s_out, s_in])

        return kernel

    @staticmethod
    def backward(ctx, grad_kernel):
        Ys, coefficients, norm_coef = ctx.saved_tensors

        grad_Ys = grad_coefficients = None

        if ctx.needs_input_grad[0]:
            grad_Ys = grad_kernel.new_zeros(*ctx.Ys_shape)  # [l_filter * m_filter, batch]
        if ctx.needs_input_grad[1]:
            grad_coefficients = grad_kernel.new_zeros(*ctx.coefficients_shape)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        begin_c = 0

        begin_out = 0
        for i, (mul_out, l_out, p_out) in enumerate(ctx.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
            begin_out += mul_out * (2 * l_out + 1)

            begin_in = 0
            for j, (mul_in, l_in, p_in) in enumerate(ctx.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
                begin_in += mul_in * (2 * l_in + 1)

                l_filters = ctx.get_l_filters(l_in, p_in, l_out, p_out)
                if not l_filters:
                    continue

                # extract the subset of the `coefficients` that corresponds to the couple (l_out, l_in)
                n = mul_out * mul_in * len(l_filters)
                if grad_Ys is not None:
                    c = coefficients[:, begin_c: begin_c + n].view(-1, mul_out, mul_in, len(l_filters))  # [batch, mul_out, mul_in, l_filter]
                if grad_coefficients is not None:
                    grad_c = grad_coefficients[:, begin_c: begin_c + n].view(-1, mul_out, mul_in, len(l_filters))  # [batch, mul_out, mul_in, l_filter]
                begin_c += n

                grad_K = grad_kernel[:, s_out, s_in].view(-1, mul_out, 2 * l_out + 1, mul_in, 2 * l_in + 1)

                n = norm_coef[i, j]  # [batch]

                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in ctx.set_of_l_filters if l < l_filter)
                    C = SO3.clebsch_gordan(l_out, l_in, l_filter, cached=True, like=grad_kernel)  # [m_out, m_in, m]

                    if grad_Ys is not None:
                        grad_Y = grad_Ys[tmp: tmp + 2 * l_filter + 1]  # [m, batch]
                        grad_Y.add_(torch.einsum("zuivj,ijk,zuv,z->kz", grad_K, C, c[..., k], n))
                    if grad_coefficients is not None:
                        Y = Ys[tmp: tmp + 2 * l_filter + 1]  # [m, batch]
                        grad_c[..., k] = torch.einsum("zuivj,ijk,kz,z->zuv", grad_K, C, Y, n)

        return grad_Ys, grad_coefficients, None, None, None, None, None
