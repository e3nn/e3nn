# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement
import math

import torch

from e3nn import o3, rs, rsh
from e3nn.linear import KernelLinear


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel,
                 selection_rule=o3.selection_rule_in_out_sh,
                 normalization='component',
                 allow_unused_inputs=False):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param selection_rule: function of signature (l_in, p_in, l_out, p_out) -> [l_filter]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.selection_rule = selection_rule
        if not allow_unused_inputs:
            self.check_input()
        self.check_output()

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

        norm_coef = torch.zeros((len(self.Rs_out), len(self.Rs_in)))

        n_path = 0
        set_of_l_filters = set()

        for i, (mul_out, l_out, p_out) in enumerate(self.Rs_out):
            # consider that we sum a bunch of [lambda_(m_out)] vectors
            # we need to count how many of them we sum in order to normalize the network
            num_summed_elements = 0
            for mul_in, l_in, p_in in self.Rs_in:
                l_filters = self.selection_rule(l_in, p_in, l_out, p_out)
                num_summed_elements += mul_in * len(l_filters)

            for j, (mul_in, l_in, p_in) in enumerate(self.Rs_in):
                # normalization assuming that each terms are of order 1 and uncorrelated
                norm_coef[i, j] = lm_normalization(l_out, l_in) / math.sqrt(num_summed_elements)

                l_filters = self.selection_rule(l_in, p_in, l_out, p_out)
                assert l_filters == sorted(set(l_filters)), "selection_rule must return a sorted list of unique values"
                if p_out != 0:
                    assert all(p_in * (-1) ** l == p_out for l in l_filters), "selection_rule must return l's compatible with SH parity"

                # compute the number of degrees of freedom
                n_path += mul_out * mul_in * len(l_filters)

                # create the set of all spherical harmonics orders needed
                set_of_l_filters = set_of_l_filters.union(l_filters)

        # create the radial model: R+ -> R^n_path
        # it contains the learned parameters
        self.R = RadialModel(n_path)
        self.set_of_l_filters = sorted(set_of_l_filters)
        self.register_buffer('norm_coef', norm_coef)

        self.linear = KernelLinear(self.Rs_in, self.Rs_out)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_input(self):
        for _, l_in, p_in in self.Rs_in:
            if not any(self.selection_rule(l_in, p_in, l_out, p_out) for _, l_out, p_out in self.Rs_out):
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def check_output(self):
        for _, l_out, p_out in self.Rs_out:
            if not any(self.selection_rule(l_in, p_in, l_out, p_out) for _, l_in, p_in in self.Rs_in):
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

    def forward(self, r, r_eps=0, custom_backward=False):
        """
        :param r: tensor [..., 3]
        :param custom_backward: call KernelFn rather than using automatic differentiation
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        *size, xyz = r.size()
        assert xyz == 3
        r = r.reshape(-1, 3)

        radii = r.norm(2, dim=1)  # [batch]

        # (1) Case r > 0

        # precompute all needed spherical harmonics
        Y = rsh.spherical_harmonics_xyz(self.set_of_l_filters, r[radii > r_eps])  # [batch, l_filter * m_filter]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        R = self.R(radii[radii > r_eps])  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        if custom_backward:
            kernel1 = KernelFn.apply(Y, R, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters)
        else:
            kernel1 = kernel_fn_forward(Y, R, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters)

        # (2) Case r = 0

        kernel2 = self.linear()

        kernel = r.new_zeros(len(r), *kernel2.shape)
        kernel[radii > r_eps] = kernel1
        kernel[radii <= r_eps] = kernel2

        return kernel.reshape(*size, *kernel2.shape)


def kernel_fn_forward(Y, R, norm_coef, Rs_in, Rs_out, selection_rule, set_of_l_filters):
    """
    :param Y: tensor [batch, l_filter * m_filter]
    :param R: tensor [batch, l_out * l_in * mul_out * mul_in * l_filter]
    :param norm_coef: tensor [l_out, l_in]
    :return: tensor [batch, l_out * mul_out * m_out, l_in * mul_in * m_in]
    """
    batch = Y.shape[0]
    n_in = rs.dim(Rs_in)
    n_out = rs.dim(Rs_out)

    kernel = Y.new_zeros(batch, n_out, n_in)

    # note: for the normalization we assume that the variance of R[i] is one
    begin_R = 0

    begin_out = 0
    for i, (mul_out, l_out, p_out) in enumerate(Rs_out):
        s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
        begin_out += mul_out * (2 * l_out + 1)

        begin_in = 0
        for j, (mul_in, l_in, p_in) in enumerate(Rs_in):
            s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
            begin_in += mul_in * (2 * l_in + 1)

            l_filters = selection_rule(l_in, p_in, l_out, p_out)
            if not l_filters:
                continue

            # extract the subset of the `R` that corresponds to the couple (l_out, l_in)
            n = mul_out * mul_in * len(l_filters)
            sub_R = R[:, begin_R: begin_R + n].reshape(batch, mul_out, mul_in, len(l_filters))  # [batch, mul_out, mul_in, l_filter]
            begin_R += n

            # note: I don't know if we can vectorize this for loop because [l_filter * m_filter] cannot be put into [l_filter, m_filter]
            K = 0
            for k, l_filter in enumerate(l_filters):
                tmp = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
                sub_Y = Y[:, tmp: tmp + 2 * l_filter + 1]  # [batch, m]

                C = o3.wigner_3j(l_out, l_in, l_filter, cached=True, like=kernel)  # [m_out, m_in, m]

                # note: The multiplication with `sub_R` could also be done outside of the for loop
                K += norm_coef[i, j] * torch.einsum("ijk,zk,zuv->zuivj", (C, sub_Y, sub_R[..., k]))  # [batch, mul_out, m_out, mul_in, m_in]

            if not isinstance(K, int):
                kernel[:, s_out, s_in] = K.reshape_as(kernel[:, s_out, s_in])
    return kernel


class KernelFn(torch.autograd.Function):
    """
    The math is presented here:
    https://slides.com/mariogeiger/backward/
    """
    @staticmethod
    def forward(ctx, Y, R, norm_coef, Rs_in, Rs_out, selection_rule, set_of_l_filters):
        f"""{kernel_fn_forward.__doc__}"""
        ctx.Rs_in = Rs_in
        ctx.Rs_out = Rs_out
        ctx.selection_rule = selection_rule
        ctx.set_of_l_filters = set_of_l_filters

        # save necessary tensors for backward
        saved_Y = saved_R = None
        if Y.requires_grad:
            ctx.Y_shape = Y.shape
            saved_R = R
        if R.requires_grad:
            ctx.R_shape = R.shape
            saved_Y = Y
        ctx.save_for_backward(saved_Y, saved_R, norm_coef)

        return kernel_fn_forward(Y, R, norm_coef, ctx.Rs_in, ctx.Rs_out, ctx.selection_rule, ctx.set_of_l_filters)

    @staticmethod
    def backward(ctx, grad_kernel):
        Y, R, norm_coef = ctx.saved_tensors

        grad_Y = grad_R = None

        if ctx.needs_input_grad[0]:
            grad_Y = grad_kernel.new_zeros(*ctx.Y_shape)  # [batch, l_filter * m_filter]
        if ctx.needs_input_grad[1]:
            grad_R = grad_kernel.new_zeros(*ctx.R_shape)  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        begin_R = 0

        begin_out = 0
        for i, (mul_out, l_out, p_out) in enumerate(ctx.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
            begin_out += mul_out * (2 * l_out + 1)

            begin_in = 0
            for j, (mul_in, l_in, p_in) in enumerate(ctx.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
                begin_in += mul_in * (2 * l_in + 1)

                l_filters = ctx.selection_rule(l_in, p_in, l_out, p_out)
                if not l_filters:
                    continue

                n = mul_out * mul_in * len(l_filters)
                if grad_Y is not None:
                    sub_R = R[:, begin_R: begin_R + n].reshape(
                        -1, mul_out, mul_in, len(l_filters)
                    )  # [batch, mul_out, mul_in, l_filter]
                if grad_R is not None:
                    sub_grad_R = grad_R[:, begin_R: begin_R + n].reshape(
                        -1, mul_out, mul_in, len(l_filters)
                    )  # [batch, mul_out, mul_in, l_filter]
                begin_R += n

                grad_K = grad_kernel[:, s_out, s_in].reshape(-1, mul_out, 2 * l_out + 1, mul_in, 2 * l_in + 1)

                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in ctx.set_of_l_filters if l < l_filter)
                    C = o3.wigner_3j(l_out, l_in, l_filter, cached=True, like=grad_kernel)  # [m_out, m_in, m]

                    if grad_Y is not None:
                        grad_Y[:, tmp: tmp + 2 * l_filter + 1] += norm_coef[i, j] * torch.einsum("zuivj,ijk,zuv->zk", grad_K, C, sub_R[..., k])
                    if grad_R is not None:
                        sub_Y = Y[:, tmp: tmp + 2 * l_filter + 1]  # [batch, m]
                        sub_grad_R[..., k] = norm_coef[i, j] * torch.einsum("zuivj,ijk,zk->zuv", grad_K, C, sub_Y)

        del ctx
        return grad_Y, grad_R, None, None, None, None, None
