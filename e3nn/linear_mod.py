# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement, unbalanced-tuple-unpacking
import torch

from e3nn import rs


def kernel_linear(Rs_in, Rs_out):
    # Compute Clebsh-Gordan coefficients
    def selection_rule(l_in, p_in, l_out, p_out):
        if l_in == l_out and p_out in [0, p_in]:
            return [0]
        return []

    Rs_f, Q = rs.tensor_product(Rs_in, selection_rule, Rs_out)  # [out, in, w]
    Rs_f = rs.simplify(Rs_f)
    [(_n_path, l, p)] = Rs_f
    assert l == 0 and p in [0, 1]

    return Q


class KernelLinear(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()

        Q = kernel_linear(Rs_in, Rs_out)  # [out, in, w]
        self.register_buffer('Q', Q)
        self.weight = torch.nn.Parameter(torch.randn(Q.shape[2]))

    def forward(self):
        """
        :return: tensor [l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        return torch.einsum('ijk,k->ij', self.Q, self.weight)


class Linear(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, allow_unused_inputs=False):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()
        self.Rs_in = rs.convention(Rs_in)
        self.Rs_out = rs.convention(Rs_out)
        if not allow_unused_inputs:
            self.check_input()
        self.check_output()

        self.kernel = KernelLinear(self.Rs_in, self.Rs_out)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def check_input(self):
        for _, l_in, p_in in self.Rs_in:
            if not any((l_in, p_in) == (l_out, p_out) for _, l_out, p_out in self.Rs_out):
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def check_output(self):
        for _, l_out, p_out in self.Rs_out:
            if not any((l_in, p_in) == (l_out, p_out) for _, l_in, p_in in self.Rs_in):
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensor [..., channel]
        """
        *size, dim_in = features.shape
        features = features.reshape(-1, dim_in)

        output = torch.einsum('ij,zj->zi', self.kernel(), features)

        return output.reshape(*size, -1)
