# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin
from torch_sparse import SparseTensor


def spreshape(sp, ncol):
    """
    reshape a sparse matrix
    """
    row, col, val = sp.coo()
    index = row * sp.size(1) + col
    return SparseTensor(
        row=index // ncol,
        col=index % ncol,
        value=val
    )


def register_sparse_buffer(module, name, sp):
    row, col, val = sp.coo()
    module.register_buffer("{}_row".format(name), row)
    module.register_buffer("{}_col".format(name), col)
    module.register_buffer("{}_val".format(name), val)


def get_sparse_buffer(module, name):
    row = getattr(module, "{}_row".format(name))
    col = getattr(module, "{}_col".format(name))
    val = getattr(module, "{}_val".format(name))
    return SparseTensor(
        row=row,
        col=col,
        value=val
    )
