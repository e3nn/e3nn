# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement, unbalanced-tuple-unpacking
import math

import torch

from e3nn import o3, rs


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, selection_rule=o3.selection_rule_in_out_sh, sh=o3.spherical_harmonics_xyz, normalization='norm'):
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
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.check_input_output(selection_rule)
        self.sh = sh

        assert isinstance(normalization, str), "normalization should be passed as a string value"
        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

        # (1) For the case r > 0

        # Compute Clebsh-Gordan coefficients
        Rs_f, Q = rs.tensor_product_in_out(self.Rs_in, self.Rs_out, selection_rule, normalization)  # [out, in, Y]

        # Sort filters representation
        Rs_f, perm = rs.sort(Rs_f)
        Rs_f = rs.simplify(Rs_f)
        Q = torch.einsum('ijk,lk->ijl', Q, perm)
        del perm

        # Get L's of the spherical harmonics
        self.set_of_l_filters = sorted({l for _mul, l, _p in Rs_f})
        assert rs.irrep_dim(Rs_f) == sum(2 * l + 1 for l in self.set_of_l_filters)

        # Normalize the spherical harmonics
        if normalization == 'component':
            diag = torch.ones(rs.irrep_dim(Rs_f))
        if normalization == 'norm':
            diag = torch.cat([torch.ones(2 * l + 1) / math.sqrt(2 * l + 1) for l in self.set_of_l_filters])
        norm_Y = math.sqrt(4 * math.pi) * torch.diag(diag)  # [Y, Y]

        # Matrix to dispatch the spherical harmonics
        mat_Y = rs.map_irrep_to_Rs(Rs_f)  # [Rs_f, Y]
        mat_Y = mat_Y @ norm_Y

        # Create the radial model: R+ -> R^n_path
        n_path = rs.mul_dim(Rs_f)
        self.R = RadialModel(n_path)

        mat_R = rs.map_mul_to_Rs(Rs_f)  # [Rs_f, R]

        mixing_matrix = torch.einsum('ijk,ky,kw->ijyw', Q, mat_Y, mat_R)
        self.register_buffer('mixing_matrix1', mixing_matrix)

        # (2) For the case r = 0

        # Compute Clebsh-Gordan coefficients
        def selection_rule_linear(l_in, p_in, l_out, p_out):
            return [0] if 0 in selection_rule(l_in, p_in, l_out, p_out) else []

        Rs_f, Q = rs.tensor_product_in_out(self.Rs_in, self.Rs_out, selection_rule_linear, normalization)  # [out, in, Y]
        Rs_f = rs.simplify(Rs_f)
        [(n_path, l, p)] = Rs_f
        assert l == 0 and p in [0, 1]

        # Create the weights
        self.weight = torch.nn.Parameter(torch.randn(n_path))

        self.register_buffer('mixing_matrix2', Q)

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
        Y = self.sh(self.set_of_l_filters, r[radii > r_eps])  # [l_filter * m_filter, batch]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        R = self.R(radii[radii > r_eps])  # [batch, l_out * l_in * mul_out * mul_in * l_filter]

        kernel1 = torch.einsum('ijyw,yz,zw->zij', self.mixing_matrix1, Y, R)

        # (2) Case r = 0

        kernel2 = torch.einsum('ijk,k->ij', self.mixing_matrix2, self.weight)

        kernel = r.new_zeros(len(r), *kernel2.shape)
        kernel[radii > r_eps] = kernel1
        kernel[radii <= r_eps] = kernel2

        return kernel.view(*size, *kernel2.shape)
