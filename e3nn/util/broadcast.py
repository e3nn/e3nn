
def broadcast_tensors(x, y, dim_x=-1, dim_y=-1):
    r"""
    Examples
    --------

    >>> import torch
    >>> x, y = broadcast_tensors(torch.randn(12), torch.randn(2, 3, 9))
    >>> assert x.shape == (2, 3, 12)
    >>> assert y.shape == (2, 3, 9)


    >>> import torch
    >>> x, y = broadcast_tensors(torch.randn(5, 3, 1, 12), torch.randn(3, 4, 9, 9), dim_y=-2)
    >>> assert x.shape == (5, 3, 4, 12)
    >>> assert y.shape == (5, 3, 4, 9, 9)
    """
    dim_x = dim_x % x.ndim
    dim_y = dim_y % y.ndim

    while dim_x < dim_y:
        x = x[None]
        dim_x += 1

    while dim_y < dim_x:
        y = y[None]
        dim_y += 1

    assert dim_x == dim_y
    dim = dim_x

    shape = [max(dx, dy) for dx, dy in zip(x.shape[:dim], y.shape[:dim])]

    x = x.expand(*shape, *(-1,)*(x.ndim - dim))
    y = y.expand(*shape, *(-1,)*(y.ndim - dim))

    return x, y
