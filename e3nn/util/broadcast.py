from typing import Tuple, List

import torch


@torch.jit.script
def broadcast_tensors(x: torch.Tensor,
                      y: torch.Tensor,
                      dim_x: int=-1,
                      dim_y: int=-1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    dim: int = dim_x

    # TorchScript requires us to build these shapes explicitly,
    # rather than using splats or list comprehensions
    xshape: List[int] = [-1] * x.ndim
    yshape: List[int] = [-1] * y.ndim
    val: int = 0
    i: int = 0
    for i in range(dim):
        val = max([x.shape[i], y.shape[i]])
        xshape[i] = val
        yshape[i] = val
    x = x.expand(xshape)
    y = y.expand(yshape)

    return x, y
