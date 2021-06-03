import warnings

import torch
from e3nn.math.group import Group


def intertwiners(group: Group, D1, D2, eps=1e-9, dtype=torch.float64, device=None):
    r"""
    Compute a basis of the vector space of matrices A such that
    D1(g) A = A D2(g) for all g in O(3)
    """
    e = group.identity(dtype=dtype, device=device)
    I1 = D1(e).contiguous()
    I2 = D2(e).contiguous()

    if I1.dtype in [torch.float16, torch.float32]:
        warnings.warn("Warning: intertwiners: you should use torch.float64")

    # picking 20 random rotations seems good enough, no idea for the finite groups
    rr = [group.random(dtype=dtype, device=device) for i in range(20)]
    xs = [torch.kron(D1(g).contiguous(), I2) - torch.kron(I1, D2(g).T.contiguous()) for g in rr]
    xtx = sum(x.T @ x for x in xs)

    res = xtx.symeig(eigenvectors=True)
    null_space = res.eigenvectors.T[res.eigenvalues.abs() < eps]
    null_space = null_space.reshape(null_space.shape[0], I1.shape[0], I2.shape[0])

    # check that it works
    solutions = []
    for A in null_space:
        d = 0
        for _ in range(4):
            g = group.random(dtype=dtype, device=device)
            d += A @ D2(g) - D1(g) @ A
        d /= 4
        if d.abs().max() < eps:
            solutions.append((d.norm(), A))
    solutions = [A for _, A in sorted(solutions, key=lambda x: x[0])]

    solutions = torch.stack(solutions) if len(solutions) > 0 else torch.zeros(0, I1.shape[0], I2.shape[0], dtype=dtype, device=device)
    solutions = torch.qr(solutions.flatten(1)).R
    return solutions.reshape(len(solutions), I1.shape[0], I2.shape[0])
