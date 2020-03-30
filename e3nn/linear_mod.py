# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import math
from functools import partial

import torch

from e3nn import rs
from e3nn import o3


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

        Rs_inter, mixing_matrix, paths = rs.tensor_product(
            self.Rs_out, self.Rs_in, partial(o3.selection_rule, lmax=0), paths=True
        )

        num_summed_list = rs.num_summed_elements(paths)
        factors = [1. / math.sqrt(num_summed / mul_out)
                   for num_summed, (mul_out, L, p) in zip(num_summed_list,
                                                          self.Rs_out)]
        norm_coef = torch.eye(len(Rs_out)) * torch.tensor(factors)
        print(norm_coef)
        full_norm_coef = torch.einsum('nm,in,jm->ij',
                                      norm_coef,
                                      rs.map_tuple_to_Rs(self.Rs_out),
                                      rs.map_tuple_to_Rs(self.Rs_in))

        self.weight = torch.nn.Parameter(torch.randn(len(paths)))
        self.register_buffer('mixing_matrix', mixing_matrix)
        self.register_buffer('norm_coef', full_norm_coef)

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

        output = torch.einsum(
            'kij,k,zj,ij->zi',
            self.mixing_matrix,
            self.weight,
            features,
            self.norm_coef
        )
        return output.view(*size, -1)
