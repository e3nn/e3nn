import pytest
import torch
from e3nn.math.group import intertwiners, has_rep_in_rep, SO3
from e3nn import o3


def test_intertwiners():
    G = SO3()
    A = torch.randn(9, 9)

    def rep(q):
        return A @ G.rep(4)(q) @ A.T

    B = intertwiners(G, G.rep(4), rep, dtype=torch.get_default_dtype())
    assert torch.allclose(A, B)


def test_has_rep_in_rep():
    if torch.get_default_dtype() != torch.float64:
        pytest.xfail('has_rep_in_rep only check out with double accuracy')

    G = SO3()
    irreps = o3.Irreps('3x0e + 2x1e')
    n, _, _ = has_rep_in_rep(G, irreps.D_from_quaternion, G.rep(1))
    assert n == 2
