# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch
from e3nn.linear import Linear
from e3nn import o3, rs


class TensorProduct(torch.nn.Module):
    """
    (A x B)_k =  C_ijk A_i B_j

    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(2, 1), (5, 0), (1, 1), (1, 2), (2, 1)]
    """
    def __init__(self, Rs_1, Rs_2, get_l_output=o3.selection_rule):
        super().__init__()

        self.get_ls = get_l_output
        self.Rs_1 = rs.simplify(Rs_1)
        self.Rs_2 = rs.simplify(Rs_2)

        Rs_out, mixing_matrix = rs.tensor_product(Rs_1, Rs_2, get_l_output=get_l_output)
        self.Rs_out = rs.simplify(Rs_out)
        self.register_buffer('mixing_matrix', mixing_matrix)

    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2

        output = torch.einsum('kij,zi,zj->zk', self.mixing_matrix, features_1,
                              features_2)
        return output.view(*size_1, -1)


class ElementwiseTensorProduct(torch.nn.Module):
    """
    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(1, 1), (1, 0), (1, 1)]
    """
    def __init__(self, Rs_1, Rs_2, get_l_output=o3.selection_rule):
        super().__init__()

        Rs_1 = rs.simplify(Rs_1)
        Rs_2 = rs.simplify(Rs_2)
        assert sum(mul for mul, _, _ in Rs_1) == sum(mul for mul, _, _ in Rs_2)

        Rs_out, mixing_matrix = rs.elementwise_tensor_product(Rs_1, Rs_2, get_l_output)
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

        output = torch.einsum('kij,zi,zj->zk', self.mixing_matrix, features_1,
                              features_2)
        return output.view(*size_1, -1)


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_mid_1, Rs_mid_2, mul_mid, Rs_out, get_l_mul=o3.selection_rule):
        super().__init__()
        self.mul_mid = mul_mid
        self.m = TensorProduct(Rs_mid_1, Rs_mid_2, get_l_mul)
        self.si = Linear(mul_mid * self.m.Rs_out, Rs_out)

    def forward(self, x1, x2):
        """
        :return:         tensor [..., channel]
        """
        # split into mul x Rs
        *size, _ = x1.shape
        x1 = x1.view(*size, self.mul_mid, -1)
        x2 = x2.view(*size, self.mul_mid, -1)

        x = self.m(x1, x2).view(*size, -1)
        x = self.si(x)
        return x
