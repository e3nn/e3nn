import torch
from e3nn.math.group import intertwiners, SO3


def test_intertwiners():
    torch.set_default_dtype(torch.float64)

    G = SO3()
    A = torch.randn(9, 9)

    def rep(q):
        return A @ G.rep(4)(q) @ A.T

    B = intertwiners(G, G.rep(4), rep, dtype=torch.get_default_dtype())
    assert torch.allclose(A, B)
