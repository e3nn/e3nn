# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
from torch_geometric.nn import MessagePassing


class E3Conv(MessagePassing):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super(E3Conv, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, x, edge_index, edge_attr, size=None, n_norm=1):
        if size is None:
            size = int(x.shape[-2])
        k = self.kernel(edge_attr)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=x, k=k)

    def message(self, x_j, k):
        out = torch.einsum('eij,ej->ei', k, x_j)
        return out

    def update(self, aggr_out):
        return aggr_out
