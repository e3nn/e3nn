# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable, abstract-method
import torch
import torch_geometric as tg
from e3nn import rsh, rs
from e3nn.tensor_product import WeightedTensorProduct


class Convolution(tg.nn.MessagePassing):
    def __init__(self, kernel):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1, groups=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_target, n_source) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k, groups=groups)

    def message(self, x_j, k, groups):
        N = x_j.shape[0]
        cout, cin = k.shape[-2:]
        x_j = x_j.view(N, groups, cin)  # Rs_tp1
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, groups * cout)
        if k.dim() == 4 and k.shape[1] == groups:  # kernel has group dimension
            return torch.einsum('egij,egj->egi', k, x_j).reshape(N, groups * cout)
        return torch.einsum('eij,egj->egi', k, x_j).reshape(N, groups * cout)


class WTPConv(tg.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, normalization='component'):
        """
        :param Rs_in:  input representation
        :param lmax:   spherical harmonic representation
        :param Rs_out: output representation
        :param RadialModel: model constructor
        """
        super().__init__(aggr='add', flow='target_to_source')
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.tp = WeightedTensorProduct(Rs_in, Rs_sh, Rs_out, normalization, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.Rs_sh = Rs_sh
        self.normalization = normalization

    def forward(self, features, edge_index, edge_r, sh=None, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param sh: Tensor of shape [num_messages, dim(Rs_sh)]
        :param size: (n_target, n_source) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        if sh is None:
            sh = rsh.spherical_harmonics_xyz(self.Rs_sh, edge_r, self.normalization)  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_r.norm(dim=1))  # [num_messages, nweight]

        return self.propagate(edge_index, size=size, x=features, sh=sh, w=w)

    def message(self, x_j, sh, w):
        """
        :param x_j: [num_messages, dim(Rs_in)]
        :param sh:  [num_messages, dim(Rs_sh)]
        :param w:   [num_messages, nweight]
        """
        return self.tp(x_j, sh, w)
