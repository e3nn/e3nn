# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch
from e3nn.linear import Linear
from e3nn import o3, rs
from functools import partial


class TensorProduct(torch.nn.Module):
    """
    (A x B)_k =  C_ijk A_i B_j

    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(2, 1), (5, 0), (1, 1), (1, 2), (2, 1)]
    """
    def __init__(self, Rs_1, Rs_2, selection_rule=o3.selection_rule):
        super().__init__()

        self.Rs_1 = rs.simplify(Rs_1)
        self.Rs_2 = rs.simplify(Rs_2)

        Rs_out, mixing_matrix = rs.tensor_product(Rs_1, Rs_2, selection_rule)
        self.Rs_out = rs.simplify(Rs_out)
        self.register_buffer('mixing_matrix', mixing_matrix)

    def __repr__(self):
        return "{name} ({Rs_1} x {Rs_2} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_1=rs.format_Rs(self.Rs_1),
            Rs_2=rs.format_Rs(self.Rs_2),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2

        output = torch.einsum('kij,zi,zj->zk', self.mixing_matrix, features_1, features_2)
        return output.view(*size_1, -1)


class ElementwiseTensorProduct(torch.nn.Module):
    """
    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(1, 1), (1, 0), (1, 1)]
    """
    def __init__(self, Rs_1, Rs_2, selection_rule=o3.selection_rule):
        super().__init__()

        Rs_1 = rs.simplify(Rs_1)
        Rs_2 = rs.simplify(Rs_2)
        assert sum(mul for mul, _, _ in Rs_1) == sum(mul for mul, _, _ in Rs_2)

        Rs_out, mixing_matrix = rs.elementwise_tensor_product(Rs_1, Rs_2, selection_rule)
        self.register_buffer("mixing_matrix", mixing_matrix)
        self.Rs_out = rs.simplify(Rs_out)

    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2

        output = torch.einsum('kij,zi,zj->zk', self.mixing_matrix, features_1, features_2)
        return output.view(*size_1, -1)


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_mid_1, Rs_mid_2, Rs_out,
                 selection_rule=o3.selection_rule, groups=1):
        super().__init__()
        self.groups = groups
        self.tp = TensorProduct(Rs_mid_1, Rs_mid_2, selection_rule)
        self.lin = Linear(groups * self.tp.Rs_out, Rs_out)

    def forward(self, x1, x2):
        """
        :return:         tensor [..., channel]
        """
        # split into mul x Rs
        *size, _ = x1.shape
        x1 = x1.view(*size, self.groups, -1)
        x2 = x2.view(*size, self.groups, -1)

        x = self.tp(x1, x2).view(*size, -1)
        x = self.lin(x)
        return x


class LearnableBispectrum(torch.nn.Module):
    def __init__(self, Rs_in, Rs_hidden, mul_out, lmax=None):
        super().__init__()
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_hidden = rs.simplify(Rs_hidden)
        if lmax is None:
            self.lmax = max(l for mul, l, _ in self.Rs_hidden)
        else:
            self.lmax = lmax
        # Learnable tensor product of signal with itself
        self.tp = LearnableTensorProduct(
            self.Rs_in, self.Rs_in, self.Rs_hidden,
            partial(o3.selection_rule, lmax=self.lmax), groups=1)
        # Dot product
        self.dot = LearnableTensorProduct(
            self.Rs_hidden, self.Rs_in, [(mul_out, 0)],
            partial(o3.selection_rule, lmax=0), groups=1)

    def forward(self, input):
        return self.dot(self.tp(input, input), input)


class LearnableMultiplicityBispectrum(torch.nn.Module):
    def __init__(self, mul_in, single_mul_Rs_in, Rs_hidden, mul_out, lmax=None):
        super().__init__()
        self.single_mul_Rs_in = rs.simplify(single_mul_Rs_in)
        self.Rs_hidden = rs.simplify(Rs_hidden)
        if lmax is None:
            self.lmax = max(l for mul, l, _ in self.Rs_hidden)
        else:
            self.lmax = lmax
        # Learnable tensor product of signal with itself
        self.tp = LearnableTensorProduct(single_mul_Rs_in,
                                         single_mul_Rs_in,
                                         self.Rs_hidden,
                                         partial(o3.selection_rule, lmax=self.lmax),
                                         groups=mul_in)
        # Dot product
        self.dot = LearnableTensorProduct(self.Rs_hidden,
                                          self.single_mul_Rs_in * mul_in,
                                          [(mul_out, 0)],
                                          partial(o3.selection_rule, lmax=0),
                                          groups=1)

    def forward(self, input):
        return self.dot(self.tp(input, input), input)
