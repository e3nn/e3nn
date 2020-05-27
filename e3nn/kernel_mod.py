# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement, unbalanced-tuple-unpacking
import math

import torch

from e3nn import o3, rs, rsh
from e3nn.linear_mod import KernelLinear


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel,
                 selection_rule=o3.selection_rule_in_out_sh,
                 normalization='component',
                 allow_unused_inputs=False,
                 allow_zero_outputs=False):
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
        if not allow_unused_inputs:
            self.check_input(selection_rule)
        if not allow_zero_outputs:
            self.check_output(selection_rule)

        self.normalization = normalization

        self.tp = rs.TensorProduct(self.Rs_in, selection_rule, Rs_out, normalization, sorted=True)
        self.Rs_f = self.tp.Rs_in2

        self.Ls = [l for _, l, _ in self.Rs_f]
        self.R = RadialModel(rs.mul_dim(self.Rs_f))

        self.linear = KernelLinear(self.Rs_in, self.Rs_out)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_output(self, selection_rule):
        for _, l_out, p_out in self.Rs_out:
            if not any(selection_rule(l_in, p_in, l_out, p_out) for _, l_in, p_in in self.Rs_in):
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

    def check_input(self, selection_rule):
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
        Y = rsh.spherical_harmonics_xyz(self.Ls, r[radii > r_eps])  # [batch, l_filter * m_filter]

        # Normalize the spherical harmonics
        if self.normalization == 'component':
            Y.mul_(math.sqrt(4 * math.pi))
        if self.normalization == 'norm':
            diag = math.sqrt(4 * math.pi) * torch.cat([torch.ones(2 * l + 1) / math.sqrt(2 * l + 1) for _, l, _ in self.Rs_f])
            Y.mul_(diag)

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        R = self.R(radii[radii > r_eps])  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        RY = rsh.mul_radial_angular(self.Rs_f, R, Y)

        if Y.shape[0] == 0:
            kernel1 = torch.zeros(0, rs.dim(self.Rs_out), rs.dim(self.Rs_in))
        else:
            kernel1 = self.tp.right(RY)

        # (2) Case r = 0

        kernel2 = self.linear()

        kernel = r.new_zeros(len(r), *kernel2.shape)
        kernel[radii > r_eps] = kernel1
        kernel[radii <= r_eps] = kernel2

        return kernel.reshape(*size, *kernel2.shape)


class FrozenKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, r, r_eps=0, selection_rule=o3.selection_rule_in_out_sh, normalization='component'):
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
        self.register_buffer('radii', r.norm(2, dim=1))  # [batch]
        self.r_eps = r_eps

        self.tp = rs.TensorProduct(self.Rs_in, selection_rule, self.Rs_out, normalization, sorted=True)
        self.Rs_f = self.tp.Rs_in2

        Y = rsh.spherical_harmonics_xyz([(1, l, p) for _, l, p in self.Rs_f], r[self.radii > self.r_eps])  # [batch, l_filter * m_filter]

        # Normalize the spherical harmonics
        if normalization == 'component':
            Y.mul_(math.sqrt(4 * math.pi))
        if normalization == 'norm':
            diag = math.sqrt(4 * math.pi) * torch.cat([torch.ones(2 * l + 1) / math.sqrt(2 * l + 1) for _, l, _ in self.Rs_f])
            Y.mul_(diag)

        self.register_buffer('Y', Y)
        self.R = RadialModel(rs.mul_dim(self.Rs_f))

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

        RY = rsh.mul_radial_angular(self.Rs_f, R, self.Y)

        if R.shape[0] == 0:
            kernel1 = torch.zeros(0, rs.dim(self.Rs_out), rs.dim(self.Rs_in))
        else:
            kernel1 = self.tp.right(RY)

        # (2) Case r = 0

        if self.linear is not None:
            kernel2 = self.linear()

            kernel = kernel1.new_zeros(len(self.radii), *kernel2.shape)
            kernel[self.radii > self.r_eps] = kernel1
            kernel[self.radii <= self.r_eps] = kernel2
        else:
            kernel = kernel1

        return kernel.reshape(*self.size, *kernel1.shape[1:])
