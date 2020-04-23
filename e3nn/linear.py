# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import math

import torch

from e3nn import rs


class KernelLinear(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        """
        super().__init__()
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        n_path = 0

        for mul_out, l_out, p_out in self.Rs_out:
            for mul_in, l_in, p_in in self.Rs_in:
                if (l_out, p_out) == (l_in, p_in):
                    # compute the number of degrees of freedom
                    n_path += mul_out * mul_in

        self.weight = torch.nn.Parameter(torch.randn(n_path))

    def forward(self):
        """
        :return: tensor [l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        kernel = self.weight.new_zeros(rs.dim(self.Rs_out), rs.dim(self.Rs_in))
        begin_w = 0

        begin_out = 0
        for mul_out, l_out, p_out in self.Rs_out:
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
            begin_out += mul_out * (2 * l_out + 1)

            n_path = 0

            begin_in = 0
            for mul_in, l_in, p_in in self.Rs_in:
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
                begin_in += mul_in * (2 * l_in + 1)

                if (l_out, p_out) == (l_in, p_in):
                    weight = self.weight[begin_w: begin_w + mul_out * mul_in].reshape(mul_out, mul_in)  # [mul_out, mul_in]
                    begin_w += mul_out * mul_in

                    eye = torch.eye(2 * l_in + 1, dtype=self.weight.dtype, device=self.weight.device)
                    kernel[s_out, s_in] = torch.einsum('uv,ij->uivj', weight, eye).reshape(mul_out * (2 * l_out + 1), mul_in * (2 * l_in + 1))
                    n_path += mul_in

            if n_path > 0:
                kernel[s_out] /= math.sqrt(n_path)

        return kernel


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
