# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin, not-callable
import torch
from torch_sparse import SparseTensor


def register_sparse_buffer(module, name, sp):
    row, col, val = sp.coo()
    module.register_buffer("{}_row".format(name), row)
    module.register_buffer("{}_col".format(name), col)
    module.register_buffer("{}_val".format(name), val)
    module.register_buffer("{}_size".format(name), torch.tensor(sp.sparse_sizes()))


def get_sparse_buffer(module, name):
    row = getattr(module, "{}_row".format(name))
    col = getattr(module, "{}_col".format(name))
    val = getattr(module, "{}_val".format(name))
    siz = getattr(module, "{}_size".format(name))
    return SparseTensor(
        row=row,
        col=col,
        value=val,
        sparse_sizes=siz.tolist(),
    )
