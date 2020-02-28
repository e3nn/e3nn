import torch

import e3nn.o3 as o3
import e3nn.rs as rs
from e3nn.kernel import Kernel


class KernelConv(Kernel):
    def __init__(self, Rs_in, Rs_out, RadialModel, get_l_filters=o3.selection_rule, sh=o3.spherical_harmonics_xyz, normalization='norm'):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param get_l_filters: function of signature (l_in, l_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super(KernelConv, self).__init__(Rs_in, Rs_out, RadialModel, get_l_filters, sh, normalization)

    def forward(self, features, geometry, mask, y=None, radii=None):
        """
        :param features: tensor [batch, b, l_in * mul_in * m_in]
        :param geometry: tensor [batch, a, b, xyz]
        :param mask:     tensor [batch, b] (In order to zero contributions from padded atoms.)
        :param y:        Optional precomputed spherical harmonics.
        :param radii:    Optional precomputed normed geometry.
        :return:         tensor [batch, a, l_out * mul_out * m_out]
        """
        batch, a, b, xyz = geometry.size()
        assert xyz == 3

        # precompute all needed spherical harmonics
        if y is None:
            y = self.sh(self.set_of_l_filters, geometry)  # [l_filter * m_filter, batch, a, b]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        if radii is None:
            radii = geometry.norm(2, dim=-1)  # [batch, a, b]
        r = self.R(radii.flatten()).reshape(
            *radii.shape, -1
        )  # [batch, a, b, l_out * l_in * mul_out * mul_in * l_filter]

        norm_coef = getattr(self, 'norm_coef')
        norm_coef = norm_coef[:, :, (radii == 0).type(torch.long)]  # [l_out, l_in, batch, a, b]

        kernel_conv = kernel_conv_automatic_backward(
            features, y, r, norm_coef, self.Rs_in, self.Rs_out, self.get_l_filters, self.set_of_l_filters
        )
        return kernel_conv * mask.unsqueeze(-1)


def kernel_conv_automatic_backward(features, Y, R, norm_coef, Rs_in, Rs_out, get_l_filters, set_of_l_filters):
    batch, a, b = Y.shape[1:]
    n_in = rs.dim(Rs_in)
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

            l_filters = get_l_filters(l_in, p_in, l_out, p_out)
            if not l_filters:
                continue

            # extract the subset of the `R` that corresponds to the couple (l_out, l_in)
            n = mul_out * mul_in * len(l_filters)
            sub_R = R[:, :, :, begin_R: begin_R + n].contiguous().view(
                batch, a, b, mul_out, mul_in, -1
            )  # [batch, a, b, mul_out, mul_in, l_filter]
            begin_R += n

            sub_norm_coef = norm_coef[i, j]  # [batch]

            K = 0
            for k, l_filter in enumerate(l_filters):
                offset = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
                sub_Y = Y[offset: offset + 2 * l_filter + 1, ...]  # [m, batch, a, b]

                C = o3.clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel_conv)  # [m_out, m_in, m]

                K += torch.einsum(
                    "ijk,kzab,zabuv,zab,zbvj->zaui",
                    C, sub_Y, sub_R[..., k], sub_norm_coef, features[..., s_in].view(batch, b, mul_in, -1)
                )  # [batch, a, mul_out, m_out]

            if K is not 0:
                kernel_conv[:, :, s_out] += K.view(batch, a, -1)

    return kernel_conv
