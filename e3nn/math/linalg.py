from typing import Tuple

import torch


def kron(*matrices):
    r"""
    Kroneker product between matrices
    """
    for m in matrices:
        assert m.dim() == 2

    if len(matrices) == 0:
        return torch.ones(1, 1)
    if len(matrices) == 1:
        return matrices[0]

    x, y, *matrices = matrices
    z = torch.einsum("ij,kl->ikjl", x, y).reshape(x.size(0) * y.size(0), x.size(1) * y.size(1))

    if matrices:
        return kron(z, *matrices)
    return z


def direct_sum(*matrices):
    r"""
    Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(*total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out


@torch.jit.script
def orthonormalize(
        vecs: torch.Tensor,
        eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param vecs: tensor of shape [n, m] with n <= m
    :return: (base, expand)
    base.shape[1] == m
    expand.shape[1] == m
    base.shape[0] + expand.shape[0] == m
    cat[base, expand] is orthonormal
    """
    assert vecs.dim() == 2
    dim = vecs.shape[1]

    base = []
    for x in vecs:
        for y in base:
            x -= torch.dot(x, y) * y
        if x.norm() > 2 * eps:
            x = x / x.norm()
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            base += [x]

    expand = []
    for x in torch.eye(dim, device=vecs.device, dtype=vecs.dtype):
        for y in base + expand:
            x -= torch.dot(x, y) * y
        if x.norm() > 2 * eps:
            x /= x.norm()
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            expand += [x]

    base = torch.stack(base) if len(base) > 0 else vecs.new_zeros(0, dim)
    expand = torch.stack(expand) if len(expand) > 0 else vecs.new_zeros(0, dim)

    return base, expand


@torch.jit.script
def complete_basis(
        vecs: torch.Tensor,
        eps: float = 1e-9
) -> torch.Tensor:
    """
    :param vecs: tensor of shape [n, m] with n <= m
    :return: (base, expand)
    base.shape[1] == m
    expand.shape[1] == m
    base.shape[0] + expand.shape[0] == m
    expand are orthonormal
    """
    assert vecs.dim() == 2
    dim = vecs.shape[1]

    base = [x for x in vecs]

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
