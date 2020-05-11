# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.linear import Linear
from e3nn.linear_mod import KernelLinear
from e3nn.util.sparse import get_sparse_buffer, register_sparse_buffer


class TensorProduct(torch.nn.Module):
    """
    (A x B)_k =  C_ijk A_i B_j

    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(2, 1), (5, 0), (1, 1), (1, 2), (2, 1)]
    """
    def __init__(self, input1, input2, output, normalization='component', sorted=True):
        super().__init__()

        Rs, mat = rs.tensor_product(input1, input2, output, normalization, sorted)

        self.Rs_in1, self.Rs_in2, self.Rs_out = input1, input2, output
        if not isinstance(self.Rs_in1, list):
            self.Rs_in1 = Rs
        if not isinstance(self.Rs_in2, list):
            self.Rs_in2 = Rs
        if not isinstance(self.Rs_out, list):
            self.Rs_out = Rs

        register_sparse_buffer(self, 'mixing_matrix', mat)

    def __repr__(self):
        return "{name} ({Rs_in1} x {Rs_in2} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in1=rs.format_Rs(self.Rs_in1),
            Rs_in2=rs.format_Rs(self.Rs_in2),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features_1, features_2):
        '''
        :param features_1: [..., channels]
        :param features_2: [..., channels]
        :return: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.reshape(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.reshape(-1, n_2)
        assert size_1 == size_2

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]

        features = torch.einsum('zi,zj->ijz', features_1, features_2)  # [in1, in2, batch]
        features = features.reshape(features_1.shape[1] * features_2.shape[1], features.shape[2])
        features = mixing_matrix @ features  # [out, batch]
        return features.T.reshape(*size_1, -1)

    def right(self, features_2):
        '''
        :param features_2: [..., ch_in2]
        :return: [..., ch_out, ch_in1]
        '''
        *size_2, n_2 = features_2.size()
        features_2 = features_2.reshape(-1, n_2)

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]
        mixing_matrix = mixing_matrix.sparse_reshape(rs.dim(self.Rs_out) * rs.dim(self.Rs_in1), rs.dim(self.Rs_in2))
        output = mixing_matrix @ features_2.T  # [out * in1, batch]
        return output.T.reshape(*size_2, rs.dim(self.Rs_out), rs.dim(self.Rs_in1))


class TensorSquare(torch.nn.Module):
    """
    (A x A)_k =  C_ijk A_i A_j

    [(2, 0), (2, 1)] x [(2, 0), (2, 1)] = [(3, 0), (4, 1), (3, 0), (1, 1), (3, 2)]
    """
    def __init__(self, Rs_in, selection_rule=o3.selection_rule):
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)

        self.Rs_out, mixing_matrix = rs.tensor_square(self.Rs_in, selection_rule, sorted=True)
        register_sparse_buffer(self, 'mixing_matrix', mixing_matrix)

    def __repr__(self):
        return "{name} ({Rs_in} ^ 2 -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        features = features.reshape(-1, n)

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")

        features = torch.einsum('zi,zj->ijz', features, features)
        features = mixing_matrix @ features.reshape(-1, features.shape[2])
        return features.T.reshape(*size, -1)


class LearnableTensorSquare(torch.nn.Module):
    """
    (A x A)_k =  C_ijk A_i A_j

    [(2, 0), (2, 1)] x [(2, 0), (2, 1)] = [(3, 0), (4, 1), (3, 0), (1, 1), (3, 2)]
    """
    def __init__(self, Rs_in, selection_rule=o3.selection_rule, mul=1):
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)
        Rs_ts, T = rs.tensor_square(self.Rs_in, selection_rule, sorted=True)
        register_sparse_buffer(self, 'T', T)  # [out, in1 * in2]

        self.Rs_out = sorted({(mul, l, p) for _, l, p in Rs_ts})
        self.kernel = KernelLinear(Rs_ts, self.Rs_out)  # [out, in, w]

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        features = features.reshape(-1, n)

        # kernel = torch.einsum('ij,jkl->ikl', self.kernel(), self.T)  # [out, in1, in2]
        T = get_sparse_buffer(self, 'T')
        kernel = (T.t() @ self.kernel().T).T.reshape(rs.dim(self.Rs_out), rs.dim(self.Rs_in), rs.dim(self.Rs_in))  # [out, in1, in2]
        features = torch.einsum('kij,zi,zj->zk', kernel, features, features)
        return features.reshape(*size, -1)


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
        register_sparse_buffer(self, "mixing_matrix", mixing_matrix)
        self.Rs_out = rs.simplify(Rs_out)

    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.reshape(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.reshape(-1, n_2)
        assert size_1 == size_2

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")

        features = torch.einsum('zi,zj->ijz', features_1, features_2)
        features = mixing_matrix @ features.reshape(-1, features.shape[2])
        return features.T.reshape(*size_1, -1)


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_mid_1, Rs_mid_2, mul_mid, Rs_out, selection_rule=o3.selection_rule):
        super().__init__()
        self.mul_mid = mul_mid
        self.tp = TensorProduct(Rs_mid_1, Rs_mid_2, selection_rule)
        self.lin = Linear(mul_mid * self.tp.Rs_out, Rs_out)

    def forward(self, x1, x2):
        """
        :return:         tensor [..., channel]
        """
        # split into mul x Rs
        *size, _ = x1.shape
        x1 = x1.reshape(*size, self.mul_mid, -1)
        x2 = x2.reshape(*size, self.mul_mid, -1)

        x = self.tp(x1, x2).reshape(*size, -1)
        x = self.lin(x)
        return x


class LearnableBispectrum(torch.nn.Module):
    def __init__(self, Rs_in, mul_hidden, mul_out):
        super().__init__()
        self.lmax = max(l for mul, l in Rs_in)
        Rs_hidden = [(mul_hidden, l) for l in range(self.lmax + 1)]
        # Learnable tensor product of signal with itself
        self.tp = LearnableTensorProduct(Rs_in, Rs_in, 1, Rs_hidden,
                                         partial(o3.selection_rule, lmax=self.lmax))
        # Dot product
        self.dot = LearnableTensorProduct(Rs_hidden, Rs_in, 1,
                                          [(mul_out, 0)], partial(o3.selection_rule, lmax=0))

    def forward(self, input):
        output = input
        output = self.tp(output, output)
        return self.dot(output, input)
