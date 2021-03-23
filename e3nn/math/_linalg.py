from typing import Tuple

import torch


def direct_sum(*matrices):
    r"""Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out


@torch.jit.script
def orthonormalize(
        original: torch.Tensor,
        eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""orthonomalize vectors

    Parameters
    ----------
    original : `torch.Tensor`
        list of the original vectors :math:`x`

    eps : float
        a small number

    Returns
    -------
    final : `torch.Tensor`
        list of orthonomalized vectors :math:`y`

    matrix : `torch.Tensor`
        the matrix :math:`A` such that :math:`y = A x`
    """
    assert original.dim() == 2
    dim = original.shape[1]

    final = []
    matrix = []

    for i, x in enumerate(original):
        cx = x.new_zeros(len(original))
        cx[i] = 1
        for j, y in enumerate(final):
            c = torch.dot(x, y)
            x = x - c * y
            cx = cx - c * matrix[j]
        if x.norm() > 2 * eps:
            c = 1 / x.norm()
            x = c * x
            cx = c * cx
            x[x.abs() < eps] = 0
            cx[cx.abs() < eps] = 0
            c = x[x.nonzero()[0, 0]].sign()
            x = c * x
            cx = c * cx
            final += [x]
            matrix += [cx]

    final = torch.stack(final) if len(final) > 0 else original.new_zeros((0, dim))
    matrix = torch.stack(matrix) if len(matrix) > 0 else original.new_zeros((0, len(original)))

    return final, matrix


@torch.jit.script
def complete_basis(
        vecs: torch.Tensor,
        eps: float = 1e-9
) -> torch.Tensor:
    assert vecs.dim() == 2
    dim = vecs.shape[1]

    base = [x / x.norm() for x in vecs]

    expand = []
    for x in torch.eye(dim, device=vecs.device, dtype=vecs.dtype):
        for y in base + expand:
            x -= torch.dot(x, y) * y
        if x.norm() > 2 * eps:
            x /= x.norm()
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            expand += [x]

    expand = torch.stack(expand) if len(expand) > 0 else vecs.new_zeros(0, dim)

    return expand
