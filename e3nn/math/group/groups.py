import itertools
import math
from functools import partial
from abc import ABC, abstractmethod

import torch
from e3nn import o3
from e3nn.math import perm


class Group(ABC):  # pragma: no cover
    @abstractmethod
    def irrep_indices(self):
        while False:
            yield None

    @abstractmethod
    def rep(self, r, dtype=None, device=None):
        return NotImplemented

    def rep_dim(self, r, dtype=None, device=None):
        return self.rep(r, dtype=dtype, device=device)(self.identity(dtype=dtype, device=device)).shape[0]

    @abstractmethod
    def compose(self, g1, g2):
        return NotImplemented

    @abstractmethod
    def random(self, dtype=None, device=None):
        return NotImplemented

    @abstractmethod
    def identity(self, dtype=None, device=None):
        return NotImplemented

    @abstractmethod
    def inverse(self, g):
        return NotImplemented


class FiniteGroup(Group):  # pragma: no cover
    @abstractmethod
    def elements(self):
        while False:
            yield None

    @abstractmethod
    def order(self) -> int:
        return NotImplemented


class LieGroup(Group):  # pragma: no cover
    @abstractmethod
    def haar(self, g):
        return NotImplemented


class Sn(FiniteGroup):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def irrep_indices(self):
        # many are missing https://en.wikipedia.org/wiki/Representation_theory_of_the_symmetric_group
        for ir in ['trivial', 'sign', 'standard', 'sign standard']:
            yield ir

    def rep(self, r, dtype=None, device=None):
        if r == 'trivial':
            def rep(p):
                return torch.ones(1, 1, dtype=dtype, device=device)
            return rep
        if r == 'sign':
            def rep(p):
                return perm.sign(p) * torch.ones(1, 1, dtype=dtype, device=device)
            return rep
        if r == 'standard':
            return partial(perm.standard_representation, dtype=dtype, device=device)
        if r == 'sign standard':
            def rep(p):
                return perm.sign(p) * perm.standard_representation(p, dtype=dtype, device=device)

            return rep
        if r == 'natural':  # pragma: no cover
            return partial(perm.natural_representation, dtype=dtype, device=device)
        if r == 'sign natural':  # pragma: no cover
            def rep(p):
                return perm.sign(p) * perm.natural_representation(p, dtype=dtype, device=device)
            return rep

    def compose(self, p1, p2):
        return perm.compose(p1, p2)

    def random(self, **_):
        return perm.rand(self.n)

    def identity(self, **_):
        return perm.identity(self.n)

    def inverse(self, p):
        return perm.inverse(p)

    def elements(self):
        for i in range(self.order()):
            yield perm.from_int(i, self.n)

    def order(self):
        return math.factorial(self.n)


class SO3(LieGroup):
    def irrep_indices(self):
        for l in itertools.count():
            yield l

    def rep(self, l, **_):
        return o3.Irrep(l, 1).D_from_quaternion

    def rep_dim(self, l, **_):
        return 2 * l + 1

    def compose(self, q1, q2):
        return o3.compose_quaternion(q1, q2)

    def random(self, dtype=None, device=None):
        return o3.rand_quaternion(dtype=None, device=None)

    def identity(self, dtype=None, device=None):
        return o3.identity_quaternion(dtype=None, device=None)

    def inverse(self, q):
        return o3.inverse_quaternion(q)

    def haar(self, q):
        _axis, angle = o3.quaternion_to_axis_angle(q)
        return angle


class O3(LieGroup):
    def irrep_indices(self):
        for l in itertools.count():
            yield o3.Irrep(l, (-1)**l)
            yield o3.Irrep(l, -(-1)**l)

    def rep(self, r):
        def rep(g):
            q, p = g
            return r.D_from_quaternion(q, p)
        return rep

    def rep_dim(self, r):
        return r.dim

    def compose(self, g1, g2):
        q1, p1 = g1
        q2, p2 = g2
        return (o3.compose_quaternion(q1, q2), (p1 + p2) % 2)

    def random(self, dtype=None, device=None):
        return (o3.rand_quaternion(dtype=dtype, device=device),
                torch.randint(2, size=(), dtype=dtype, device=device))

    def identity(self, dtype=None, device=None):
        return (o3.identity_quaternion(dtype=dtype, device=device),
                torch.tensor(0, dtype=dtype, device=device))

    def inverse(self, g):
        q, k = g
        return (o3.inverse_quaternion(q), k)

    def haar(self, g):
        q, k = g
        _axis, angle = o3.quaternion_to_axis_angle(q)
        if k % 2 == 0:
            return angle
        else:
            return math.inf


def is_representation(group: LieGroup, D, eps, dtype=None, device=None):
    e = group.identity(dtype=dtype, device=device)
    I = D(e)
    if not torch.allclose(I, torch.eye(len(I), dtype=dtype, device=device)):
        return False

    for _ in range(4):
        g1 = group.random(dtype=dtype, device=device)
        g2 = group.random(dtype=dtype, device=device)

        g12 = group.compose(g1, g2)
        D12 = D(g12)

        D1D2 = D(g1) @ D(g2)

        if (D12 - D1D2).abs().max().item() > eps * D12.abs().max().item():
            return False
    return True


def is_group(g: LieGroup, eps) -> bool:
    e = g.identity()
    g1 = g.random()
    g2 = g.random()
    g3 = g.random()

    g4 = g.compose(e, g1)
    if not g.haar(g.compose(g4, g.inverse(g1))) < eps:
        return False

    g4 = g.compose(g.compose(g1, g2), g3)
    g5 = g.compose(g1, g.compose(g2, g3))
    if not g.haar(g.compose(g4, g.inverse(g5))) < eps:
        return False

    return True
