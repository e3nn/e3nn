import math

import pytest

from e3nn.math import perm


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_inverse(n):
    for p in perm.group(n):
        ip = perm.inverse(p)

        assert perm.compose(p, ip) == perm.identity(n)
        assert perm.compose(ip, p) == perm.identity(n)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_int_inverse(n):
    for j in range(math.factorial(n)):
        p = perm.from_int(j, n)
        i = perm.to_int(p)
        assert i == j


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_int_injection(n):
    group = {perm.from_int(j, n) for j in range(math.factorial(n))}
    assert len(group) == math.factorial(n)


def test_germinate():
    assert perm.is_group(perm.germinate({(1, 2, 3, 4, 0)}))
    assert perm.is_group(perm.germinate({(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)}))


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_rand(n):
    perm.is_perm(perm.rand(n))


def test_not_group():
    assert not perm.is_group(set())  # empty
    assert not perm.is_group({(1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)})  # missing neutral
    assert not perm.is_group({(0, 1, 2), (1, 2, 0)})  # missing inverse
