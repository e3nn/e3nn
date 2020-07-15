# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric as tg

from e3nn import rs
from e3nn.linear_mod import KernelLinear


class Convolution(tg.nn.MessagePassing):
    def __init__(self, kernel):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_source, n_target) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k)

    def message(self, x_j, k):
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, k.shape[1])
        return torch.einsum('eij,ej->ei', k, x_j)


class SmallConvolution(tg.nn.MessagePassing):
    def __init__(self, Kernel, Rs_in, Rs_out, lin_mul, tp_mul):
        super(SmallConvolution, self).__init__(
            aggr='add', flow='target_to_source')
        if lin_mul % tp_mul != 0:
            raise ValueError("linear_mul must be divisable" +
                             " by tensor_product_mul but " + f"{lin_mul} % {tp_mul} = {lin_mul % tp_mul}")
        self.lin_mul = lin_mul
        self.tp_mul = tp_mul

        self.Rs_in = rs.convention(Rs_in)
        self.Rs_out = rs.convention(Rs_out)

        lps = [(l, p) for m, l, p in self.Rs_in]
        self.Rs_lin1 = [(1, l, p) for i in range(lin_mul) for l, p in lps]
        self.Rs_tp1 = [(1, l, p) for i in range(tp_mul) for l, p in lps]

        lps = [(l, p) for m, l, p in self.Rs_out]
        self.Rs_lin2 = [(1, l, p) for i in range(lin_mul) for l, p in lps]
        self.Rs_tp2 = [(1, l, p) for i in range(tp_mul) for l, p in lps]

        self.lin1 = KernelLinear(Rs_in, self.Rs_lin1)
        self.kernel = Kernel(self.Rs_tp1, self.Rs_tp2)
        self.lin2 = KernelLinear(self.Rs_lin2, Rs_out)

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_source, n_target) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        x = features
        if size is None:
            N = x.shape[0]
            size = (N, N)
        x = (self.lin1() @ x.transpose(1, 0)).transpose(
            1, 0)  # Rs_in -> Rs_lin1
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)

        x = self.propagate(edge_index, size=size, x=x, k=k)  # Rs_tp1 -> Rs_tp2
        x = (self.lin2() @ x.transpose(1, 0)).transpose(
            1, 0)  # Rs_lin2 -> Rs_out

        return x

    def message(self, x_j, k):
        N = x_j.shape[0]
        cout, cin = k.shape[-2:]
        x_j = x_j.view(N, self.lin_mul // self.tp_mul, cin)  # Rs_tp1
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, (self.lin_mul // self.tp_mul) * cout)
        return torch.einsum('eij,egj->egi', k, x_j).reshape(N, self.lin_mul // self.tp_mul * cout)
