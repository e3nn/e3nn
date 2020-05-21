# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric as tg


class MessagePassing(tg.nn.MessagePassing):
    def __init__(self, kernel):
        super(MessagePassing, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_source, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = targets
                           edge_index[1] = sources
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_source - position_target
        :param size: (n_target, n_source) or None
        :param n_norm: typical number of sources per target

        :return: Tensor of shape [n_target, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k)

    def message(self, x_j, k):
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, k.shape[1])
        return torch.einsum('eij,ej->ei', k, x_j)
