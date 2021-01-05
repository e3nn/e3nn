import itertools
import math
from abc import ABC, abstractmethod

import torch
from e3nn import o3


class LieGroup(ABC):  # pragma: no cover
    @abstractmethod
    def irrep_indices(self):
        while False:
            yield None

    @abstractmethod
    def irrep(self, r):
        return NotImplemented

    def irrep_dim(self, r):
        return self.irrep(r)(self.identity()).shape[0]

    @abstractmethod
    def compose(self, g1, g2):
        return NotImplemented

    @abstractmethod
    def random(self):
        return NotImplemented

    @abstractmethod
    def identity(self):
        return NotImplemented

    @abstractmethod
    def inverse(self, g):
        return NotImplemented

    @abstractmethod
    def haar(self, g):
        return NotImplemented


class SO3(LieGroup):
    def irrep_indices(self):
        for l in itertools.count():
            yield l

    def irrep(self, l):
        return o3.Irrep(l, 1).D_from_quaternion

    def irrep_dim(self, l):
        return 2 * l + 1

    def compose(self, q1, q2):
        return o3.compose_quaternion(q1, q2)

    def random(self):
        return o3.rand_quaternion()

    def identity(self):
        return o3.identity_quaternion()

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

    def irrep(self, r):
        def rep(g):
            q, p = g
            return r.D_from_quaternion(q, p)
        return rep

    def irrep_dim(self, r):
        return r.dim

    def compose(self, g1, g2):
        q1, p1 = g1
        q2, p2 = g2
        return (o3.compose_quaternion(q1, q2), (p1 + p2) % 2)

    def random(self):
        return (o3.rand_quaternion(), torch.randint(2, size=()))

    def identity(self):
        return (o3.identity_quaternion(), torch.tensor(0))

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


def is_representation(group: LieGroup, D, eps):
    e = group.identity()
    I = D(e)

    if not torch.allclose(I, torch.eye(len(I), dtype=I.dtype)):
        return False

    for _ in range(4):
        g1 = group.random()
        g2 = group.random()

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
