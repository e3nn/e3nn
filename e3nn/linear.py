# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import math

import torch

from e3nn import rs


class Linear(torch.nn.Module):
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

        self.check_input_output()

        n_path = 0

        for mul_out, l_out, p_out in self.Rs_out:
            for mul_in, l_in, p_in in self.Rs_in:
                if (l_out, p_out) == (l_in, p_in):
                    # compute the number of degrees of freedom
                    n_path += mul_out * mul_in

        self.weight = torch.nn.Parameter(torch.randn(n_path))

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
                if (l_in, p_in) == (l_out, p_out):
                    has_path = True
                    break
            if not has_path:
                raise ValueError("warning! the output (l={}, p={}) cannot be generated".format(l_out, p_out))

        for _, l_in, p_in in self.Rs_in:
            has_path = False
            for _, l_out, p_out in self.Rs_out:
                if (l_in, p_in) == (l_out, p_out):
                    has_path = True
                    break
            if not has_path:
                raise ValueError("warning! the input (l={}, p={}) cannot be used".format(l_in, p_in))

    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensor [..., channel]
        """
        *size, dim_in = features.shape
        features = features.view(-1, dim_in)

        output = features.new_zeros(features.shape[0], rs.dim(self.Rs_out))
        begin_w = 0

        begin_out = 0
        for mul_out, l_out, p_out in self.Rs_out:
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
            begin_out += mul_out * (2 * l_out + 1)

            n_path = 0
            out = 0  # [batch, mul_out, m_out]

            begin_in = 0
            for mul_in, l_in, p_in in self.Rs_in:
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
                begin_in += mul_in * (2 * l_in + 1)

                if (l_out, p_out) == (l_in, p_in):
                    weight = self.weight[begin_w: begin_w + mul_out * mul_in].reshape(mul_out, mul_in)  # [mul_out, mul_in]
                    begin_w += mul_out * mul_in

                    f = features[:, s_in].view(-1, mul_in, 2 * l_in + 1)  # [batch, mul_in, m_in]
                    out += torch.einsum('uv,zvi->zui', weight, f)  # [batch, mul_out, m_out]
                    n_path += mul_in

            output[:, s_out] = out.reshape(-1, mul_out * (2 * l_out + 1)) / math.sqrt(n_path)

        return output.view(*size, -1)
