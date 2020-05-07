import torch
from torch_geometric.nn import MessagePassing
from e3nn.kernel_mod import kernel_geometric


class E3Conv(MessagePassing):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super(E3Conv, self).__init__(aggr='add')
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, x, edge_index, edge_attr, size=None, n_norm=1):
        if size is None:
            size = edge_index.shape
        k = self.kernel(edge_attr)
        print('kernel forward: ', k.shape)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=x, k=k)

    def message(self, x_i, k):
        print('x_i: ', x_i.shape, x_i.dtype)
        print('k: ', k.shape, k.dtype)
        out = torch.einsum('eij,ej->ei', k, x_i)
        print(out.shape)
        return out

    def update(self, aggr_out):
        return aggr_out