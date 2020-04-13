# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement, unbalanced-tuple-unpacking
import math

import torch

from e3nn import o3, rs, rsh
from e3nn.linear_mod import KernelLinear


def kernel_geometric(Rs_in, Rs_out, selection_rule=o3.selection_rule_in_out_sh, normalization='component'):
    # Compute Clebsh-Gordan coefficients
    Rs_f, Q = rs.tensor_product(Rs_in, selection_rule, Rs_out, normalization)  # [out, in, Y]

    # Sort filters representation
    Rs_f, perm = rs.sort(Rs_f)
    Rs_f = rs.simplify(Rs_f)
    Q = torch.einsum('ijk,lk->ijl', Q, perm)
    del perm

    # Normalize the spherical harmonics
    if normalization == 'component':
        diag = torch.ones(rs.irrep_dim(Rs_f))
    if normalization == 'norm':
        diag = torch.cat([torch.ones(2 * l + 1) / math.sqrt(2 * l + 1) for _, l, _ in Rs_f])
    norm_Y = math.sqrt(4 * math.pi) * torch.diag(diag)  # [Y, Y]

    # Matrix to dispatch the spherical harmonics
    mat_Y = rs.map_irrep_to_Rs(Rs_f)  # [Rs_f, Y]
    mat_Y = mat_Y @ norm_Y

    # Create the radial model: R+ -> R^n_path
    mat_R = rs.map_mul_to_Rs(Rs_f)  # [Rs_f, R]

    mixing_matrix = torch.einsum('ijk,ky,kw->ijyw', Q, mat_Y, mat_R)  # [out, in, Y, R]
    return Rs_f, mixing_matrix


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, selection_rule=o3.selection_rule_in_out_sh, sh=rsh.spherical_harmonics_xyz, normalization='component'):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param selection_rule: function of signature (l_in, p_in, l_out, p_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()

        self.Rs_in = rs.convention(Rs_in)
        self.Rs_out = rs.convention(Rs_out)
        self.check_input_output(selection_rule)

        Rs_f, Q = kernel_geometric(self.Rs_in, self.Rs_out, selection_rule, normalization)
        self.register_buffer('Q', Q)  # [out, in, Y, R]

        self.sh = sh
        self.Ls = [l for _, l, _ in Rs_f]
        self.R = RadialModel(rs.mul_dim(Rs_f))

        self.linear = KernelLinear(self.Rs_in, self.Rs_out)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_input_output(self, selection_rule):
        for _, l_out, p_out in self.Rs_out:
            if not any(selection_rule(l_in, p_in, l_out, p_out) for _, l_in, p_in in self.Rs_in):
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

        for _, l_in, p_in in self.Rs_in:
            if not any(selection_rule(l_in, p_in, l_out, p_out) for _, l_out, p_out in self.Rs_out):
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def forward(self, r, r_eps=0, **_kwargs):
        """
        :param r: tensor [..., 3]
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        *size, xyz = r.size()
        assert xyz == 3
        r = r.reshape(-1, 3)

        radii = r.norm(2, dim=1)  # [batch]

        # (1) Case r > 0

        # precompute all needed spherical harmonics
        Y = self.sh(self.Ls, r[radii > r_eps])  # [batch, l_filter * m_filter]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        R = self.R(radii[radii > r_eps])  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        kernel1 = torch.einsum('ijyw,zy,zw->zij', self.Q, Y, R)

        # (2) Case r = 0

        kernel2 = self.linear()

        kernel = r.new_zeros(len(r), *kernel2.shape)
        kernel[radii > r_eps] = kernel1
        kernel[radii <= r_eps] = kernel2

        return kernel.view(*size, *kernel2.shape)


class FrozenKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, r, r_eps=0, selection_rule=o3.selection_rule_in_out_sh, sh=rsh.spherical_harmonics_xyz, normalization='component'):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param tensor r: [..., 3]
        :param float r_eps: distance considered as zero
        :param selection_rule: function of signature (l_in, p_in, l_out, p_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()

        self.Rs_in = rs.convention(Rs_in)
        self.Rs_out = rs.convention(Rs_out)
        self.check_input_output(selection_rule)

        *self.size, xyz = r.size()
        assert xyz == 3
        r = r.reshape(-1, 3)  # [batch, space]
        self.radii = r.norm(2, dim=1)  # [batch]
        self.r_eps = r_eps

        Rs_f, Q = kernel_geometric(self.Rs_in, self.Rs_out, selection_rule, normalization)
        Y = sh([l for _, l, _ in Rs_f], r[self.radii > self.r_eps])  # [batch, l_filter * m_filter]
        Q = torch.einsum('ijyw,zy->zijw', Q, Y)
        self.register_buffer('Q', Q)  # [out, in, Y, R]

        self.R = RadialModel(rs.mul_dim(Rs_f))

        if (self.radii <= self.r_eps).any():
            self.linear = KernelLinear(self.Rs_in, self.Rs_out)
        else:
            self.linear = None

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_input_output(self, selection_rule):
        for _, l_out, p_out in self.Rs_out:
            if not any(selection_rule(l_in, p_in, l_out, p_out) for _, l_in, p_in in self.Rs_in):
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

        for _, l_in, p_in in self.Rs_in:
            if not any(selection_rule(l_in, p_in, l_out, p_out) for _, l_out, p_out in self.Rs_out):
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def forward(self):
        """
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        # (1) Case r > 0

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        R = self.R(self.radii[self.radii > self.r_eps])  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        kernel1 = torch.einsum('zijw,zw->zij', self.Q, R)

        # (2) Case r = 0

        if self.linear is not None:
            kernel2 = self.linear()

            kernel = kernel1.new_zeros(len(self.radii), *kernel2.shape)
            kernel[self.radii > self.r_eps] = kernel1
            kernel[self.radii <= self.r_eps] = kernel2
        else:
            kernel = kernel1

        return kernel.view(*self.size, *kernel2.shape)
