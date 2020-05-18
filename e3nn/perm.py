# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from typing import Tuple, Set
import random
import math

TY_PERM = Tuple[int]


def is_perm(p: TY_PERM):
    return sorted(set(p)) == list(range(len(p)))


def identity(n: int) -> TY_PERM:
    return tuple(i for i in range(n))


def compose(p1: TY_PERM, p2: TY_PERM) -> TY_PERM:
    """
    compute p1 . p2
    """
    assert is_perm(p1) and is_perm(p2)
    assert len(p1) == len(p2)
    # p: i |-> p[i]

    # [p1.p2](i) = p1(p2(i)) = p1[p2[i]]
    return tuple(p1[p2[i]] for i in range(len(p1)))


def inverse(p: TY_PERM) -> TY_PERM:
    """
    compute the inverse permutation
    """
    return tuple(p.index(i) for i in range(len(p)))


def rand(n: int) -> TY_PERM:
    i = random.randint(0, math.factorial(n) - 1)
    return from_int(i, n)


def from_int(i: int, n: int) -> TY_PERM:
    pool = list(range(n))
    p = []
    for _ in range(n):
        j = i % n
        i = i // n
        p.append(pool.pop(j))
        n -= 1
    return tuple(p)


def to_int(p: TY_PERM) -> int:
    n = len(p)
    pool = list(range(n))
    i = 0
    m = 1
    for j in p:
        k = pool.index(j)
        i += k * m
        m *= len(pool)
        pool.pop(k)
    return i


def group(n: int) -> Set[TY_PERM]:
    return {from_int(i, n) for i in range(math.factorial(n))}


def germinate(subset: Set[TY_PERM]) -> Set[TY_PERM]:
    while True:
        n = len(subset)
        subset = subset.union([inverse(p) for p in subset])
        subset = subset.union([
            compose(p1, p2)
            for p1 in subset
            for p2 in subset
        ])
        if len(subset) == n:
            return subset


def is_group(g: Set[TY_PERM]) -> bool:
    if len(g) == 0:
        return False

    n = len(next(iter(g)))

    for p in g:
        assert len(p) == n, p

    if not identity(n) in g:
        return False

    for p in g:
        if not inverse(p) in g:
            return False

    for p1 in g:
        for p2 in g:
            if not compose(p1, p2) in g:
                return False

    return True
