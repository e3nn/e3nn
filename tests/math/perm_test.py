import math

import pytest
import torch
from e3nn.math import perm


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_inverse(n) -> None:
    for p in perm.group(n):
        ip = perm.inverse(p)

        assert perm.compose(p, ip) == perm.identity(n)
        assert perm.compose(ip, p) == perm.identity(n)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_int_inverse(n) -> None:
    for j in range(math.factorial(n)):
        p = perm.from_int(j, n)
        i = perm.to_int(p)
        assert i == j


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_int_injection(n) -> None:
    group = {perm.from_int(j, n) for j in range(math.factorial(n))}
    assert len(group) == math.factorial(n)


def test_germinate() -> None:
    assert perm.is_group(perm.germinate({(1, 2, 3, 4, 0)}))
    assert perm.is_group(perm.germinate({(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)}))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_rand(n) -> None:
    perm.is_perm(perm.rand(n))


def test_not_group() -> None:
    assert not perm.is_group(set())  # empty
    assert not perm.is_group({(1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)})  # missing neutral
    assert not perm.is_group({(0, 1, 2), (1, 2, 0)})  # missing inverse
    assert not perm.is_group({(0, 1, 2, 3), (3, 0, 1, 2), (1, 2, 3, 0)})  # g1 . g2 not in G


def test_to_cycles() -> None:
    assert perm.to_cycles((1, 2, 3, 0)) == {(0, 1, 2, 3)}
    assert perm.to_cycles((2, 3, 0, 1)) == {(0, 2), (1, 3)}


def test_sign() -> None:
    assert perm.sign((1, 0, 3, 2)) == 1
    assert perm.sign((1, 0, 3, 2, 5, 6, 7, 4)) == -1


@pytest.mark.parametrize("n", [3, 7, 15])
def test_standard_representation(float_tolerance, n) -> None:
    # identity
    e = perm.standard_representation(perm.identity(n))
    assert torch.allclose(e, torch.eye(n - 1), atol=float_tolerance)

    # inverse
    p = perm.rand(n)
    a = perm.standard_representation(p)
    b = perm.standard_representation(perm.inverse(p))
    assert torch.allclose(a, torch.inverse(b), atol=float_tolerance)

    # compose
    p1, p2 = perm.rand(n), perm.rand(n)
    a = perm.standard_representation(p1) @ perm.standard_representation(p2)
    b = perm.standard_representation(perm.compose(p1, p2))
    assert torch.allclose(a, b, atol=float_tolerance)

    # orthogonal
    a = perm.standard_representation(perm.rand(n))
    assert torch.allclose(a @ a.T, torch.eye(n - 1), atol=float_tolerance)


@pytest.mark.parametrize("n", [3, 7, 15])
def test_natural_representation(float_tolerance, n) -> None:
    p = perm.rand(n)
    a = torch.eye(n)[list(perm.inverse(p))]
    b = perm.natural_representation(p)
    assert torch.allclose(a, b, atol=float_tolerance)

    p = perm.rand(n)
    a = torch.eye(n)[:, list(p)]
    b = perm.natural_representation(p)
    assert torch.allclose(a, b, atol=float_tolerance)

    # orthogonal
    a = perm.natural_representation(perm.rand(n))
    assert torch.allclose(a @ a.T, torch.eye(n), atol=float_tolerance)
