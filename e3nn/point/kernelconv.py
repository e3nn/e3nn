# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement
import torch

from e3nn import o3, rs, rsh
from e3nn.kernel import Kernel


class KernelConv(Kernel):
    def forward(self, features, difference_geometry, mask, y=None, radii=None, custom_backward=True):
        """
        :param features: tensor [batch, b, l_in * mul_in * m_in]
        :param difference_geometry: tensor [batch, a, b, xyz]
        :param mask:     tensor [batch, a] (In order to zero contributions from padded atoms.)
        :param y:        Optional precomputed spherical harmonics.
        :param radii:    Optional precomputed normed geometry.
        :param custom_backward: call KernelConvFn rather than using automatic differentiation, (default True)
        :return:         tensor [batch, a, l_out * mul_out * m_out]
        """
        _batch, _a, _b, xyz = difference_geometry.size()
        assert xyz == 3

        if radii is None:
            radii = difference_geometry.norm(2, dim=-1)  # [batch, a, b]

        # precompute all needed spherical harmonics
        if y is None:
            y = rsh.spherical_harmonics_xyz(self.set_of_l_filters, difference_geometry)  # [batch, a, b, l_filter * m_filter]

        y[radii == 0] = 0

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        r = self.R(radii.flatten()).reshape(*radii.shape, -1)  # [batch, a, b, l_out * l_in * mul_out * mul_in * l_filter]
        r = r.clone()
        r[radii == 0] = 0

        if custom_backward:
            output = KernelConvFn.apply(
                features, y, r, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters
            )
        else:
            output = kernel_conv_fn_forward(
                features, y, r, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters
            )

        # Case r > 0
        if radii.shape[1] == radii.shape[2]:
            output += torch.einsum('ij,zaj->zai', self.linear(), features)

        return output * mask.unsqueeze(-1)


def kernel_conv_fn_forward(F, Y, R, norm_coef, Rs_in, Rs_out, selection_rule, set_of_l_filters):
    """
    :param F: tensor [batch, b, l_in * mul_in * m_in]
    :param Y: tensor [l_filter * m_filter, batch, a, b]
    :param R: tensor [batch, a, b, l_out * l_in * mul_out * mul_in * l_filter]
    :param norm_coef: tensor [l_out, l_in]
    :return: tensor [batch, a, l_out * mul_out * m_out, l_in * mul_in * m_in]
    """
    batch, a, b, _ = Y.shape
    n_out = rs.dim(Rs_out)

    kernel_conv = Y.new_zeros(batch, a, n_out)

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
            sub_R = R[:, :, :, begin_R: begin_R + n].reshape(
                batch, a, b, mul_out, mul_in, -1
            )  # [batch, a, b, mul_out, mul_in, l_filter]
            begin_R += n

            K = 0
            for k, l_filter in enumerate(l_filters):
                offset = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
                sub_Y = Y[..., offset: offset + 2 * l_filter + 1]  # [batch, a, b, m]

                C = o3.wigner_3j(l_out, l_in, l_filter, cached=True, like=kernel_conv)  # [m_out, m_in, m]

                K += norm_coef[i, j] * torch.einsum(
                    "ijk,zabk,zabuv,zbvj->zaui",
                    C, sub_Y, sub_R[..., k], F[..., s_in].reshape(batch, b, mul_in, -1)
                )  # [batch, a, mul_out, m_out]

            if not isinstance(K, int):
                kernel_conv[:, :, s_out] += K.reshape(batch, a, -1)

    return kernel_conv


class KernelConvFn(torch.autograd.Function):
    """
    The math is presented here:
    https://slides.com/bkmi/convolution-and-kernel/
    """
    @staticmethod
    def forward(ctx, F, Y, R, norm_coef, Rs_in, Rs_out, selection_rule, set_of_l_filters):
        f"""{kernel_conv_fn_forward.__doc__}"""
        ctx.batch, ctx.a, ctx.b, _ = Y.shape
        ctx.Rs_in = Rs_in
        ctx.Rs_out = Rs_out
        ctx.selection_rule = selection_rule
        ctx.set_of_l_filters = set_of_l_filters

        # save necessary tensors for backward
        saved_Y = saved_R = saved_F = None
        if F.requires_grad:
            ctx.F_shape = F.shape
            saved_R = R
            saved_Y = Y
        if Y.requires_grad:
            ctx.Y_shape = Y.shape
            saved_R = R
            saved_F = F
        if R.requires_grad:
            ctx.R_shape = R.shape
            saved_Y = Y
            saved_F = F
        ctx.save_for_backward(saved_F, saved_Y, saved_R, norm_coef)

        return kernel_conv_fn_forward(
            F, Y, R, norm_coef, ctx.Rs_in, ctx.Rs_out, ctx.selection_rule, ctx.set_of_l_filters
        )

    @staticmethod
    def backward(ctx, grad_kernel):  # pragma: no cover
        F, Y, R, norm_coef = ctx.saved_tensors
        batch, a, b = ctx.batch, ctx.a, ctx.b

        grad_F = grad_Y = grad_R = None

        if ctx.needs_input_grad[0]:
            grad_F = grad_kernel.new_zeros(*ctx.F_shape)  # [batch, b, l_in * mul_in * m_in]
        if ctx.needs_input_grad[1]:
            grad_Y = grad_kernel.new_zeros(*ctx.Y_shape)  # [l_filter * m_filter, batch, a, b]
        if ctx.needs_input_grad[2]:
            grad_R = grad_kernel.new_zeros(*ctx.R_shape)  # [batch, a, b, l_out * l_in * mul_out * mul_in * l_filter]

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
                if (grad_Y is not None) or (grad_F is not None):
                    sub_R = R[:, :, :, begin_R: begin_R + n].reshape(
                        batch, a, b, mul_out, mul_in, -1
                    )  # [batch, a, b, mul_out, mul_in, l_filter]
                if grad_R is not None:
                    sub_grad_R = grad_R[:, :, :, begin_R: begin_R + n].clone().reshape(
                        batch, a, b, mul_out, mul_in, -1
                    )  # [batch, a, b, mul_out, mul_in, l_filter]

                if grad_F is not None:
                    sub_grad_F = grad_F[:, :, s_in].clone().reshape(
                        batch, b, mul_in, 2 * l_in + 1
                    )  # [batch, b, mul_in, 2 * l_in + 1]
                if (grad_Y is not None) or (grad_R is not None):
                    sub_F = F[..., s_in].reshape(batch, b, mul_in, 2 * l_in + 1)

                grad_K = grad_kernel[:, :, s_out].reshape(
                    batch, a, mul_out, 2 * l_out + 1
                )

                for k, l_filter in enumerate(l_filters):
                    tmp = sum(2 * l + 1 for l in ctx.set_of_l_filters if l < l_filter)
                    C = o3.wigner_3j(l_out, l_in, l_filter, cached=True, like=grad_kernel)  # [m_out, m_in, m]

                    if (grad_F is not None) or (grad_R is not None):
                        sub_Y = Y[:, :, :, tmp: tmp + 2 * l_filter + 1]  # [batch, a, b, m]

                    if grad_F is not None:
                        sub_grad_F += norm_coef[i, j] * torch.einsum(
                            "zaui,ijk,zabk,zabuv->zbvj",
                            grad_K, C, sub_Y, sub_R[..., k]
                        )  # [batch, b, mul_in, 2 * l_in + 1
                    if grad_Y is not None:
                        grad_Y[..., tmp: tmp + 2 * l_filter + 1] += norm_coef[i, j] * torch.einsum(
                            "zaui,ijk,zabuv,zbvj->zabk",
                            grad_K, C, sub_R[..., k], sub_F
                        )  # [m, batch, a, b]
                    if grad_R is not None:
                        sub_grad_R[..., k] = norm_coef[i, j] * torch.einsum(
                            "zaui,ijk,zabk,zbvj->zabuv",
                            grad_K, C, sub_Y, sub_F
                        )  # [batch, a, b, mul_out, mul_in]
                if grad_F is not None:
                    grad_F[:, :, s_in] = sub_grad_F.reshape(batch, b, mul_in * (2 * l_in + 1))
                if grad_R is not None:
                    grad_R[..., begin_R: begin_R + n] += sub_grad_R.reshape(batch, a, b, -1)
                begin_R += n

        return grad_F, grad_Y, grad_R, None, None, None, None, None
