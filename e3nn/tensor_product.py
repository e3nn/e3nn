# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.linear import Linear
from e3nn.linear_mod import KernelLinear
from e3nn.util.sparse import get_sparse_buffer, register_sparse_buffer


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


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_mid_1, Rs_mid_2, mul_mid, Rs_out, selection_rule=o3.selection_rule):
        super().__init__()
        self.mul_mid = mul_mid
        self.tp = rs.TensorProduct(Rs_mid_1, Rs_mid_2, selection_rule)
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
