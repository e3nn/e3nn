import random

import pytest
import torch
from e3nn import o3
from e3nn.o3 import TensorProduct


def make_tp(l1, p1, l2, p2, lo, po, mode, weight):
    def mul_out(mul):
        if mode == "uvuv":
            return mul**2
        return mul

    try:
        return TensorProduct(
            [(25, (l1, p1)), (19, (l1, p1))],
            [(25, (l2, p2)), (19, (l2, p2))],
            [(mul_out(25), (lo, po)), (mul_out(19), (lo, po))],
            [
                (0, 0, 0, mode, weight),
                (1, 1, 1, mode, weight),
                (0, 0, 1, 'uvw', True, 0.5),
                (0, 1, 1, 'uvw', True, 0.2),
            ]
        )
    except AssertionError:
        return None


def random_params():
    params = set()
    while len(params) < 25:
        l1 = random.randint(0, 2)
        p1 = random.choice([-1, 1])
        l2 = random.randint(0, 2)
        p2 = random.choice([-1, 1])
        lo = random.randint(0, 2)
        po = random.choice([-1, 1])
        mode = random.choice(['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'])
        weight = random.choice([True, False])
        if make_tp(l1, p1, l2, p2, lo, po, mode, weight) is not None:
            params.add((l1, p1, l2, p2, lo, po, mode, weight))
    return params


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params())
def test(float_tolerance, assert_equivariant, l1, p1, l2, p2, lo, po, mode, weight):
    eps = float_tolerance
    n = 1_500
    tol = 3.0

    m = make_tp(l1, p1, l2, p2, lo, po, mode, weight)

    # bilinear
    x1 = torch.randn(2, m.irreps_in1.dim)
    x2 = torch.randn(2, m.irreps_in1.dim)
    y1 = torch.randn(2, m.irreps_in2.dim)
    y2 = torch.randn(2, m.irreps_in2.dim)

    z1 = m(x1 + x2, y1 + y2)
    z2 = m(x1, y1 + y2) + m(x2, y1 + y2)
    z3 = m(x1 + x2, y1) + m(x1 + x2, y2)
    assert (z1 - z2).abs().max() < eps
    assert (z1 - z3).abs().max() < eps

    # right
    z1 = m(x1, y1)
    z2 = torch.einsum('zi,zij->zj', x1, m.right(y1))
    assert (z1 - z2).abs().max() < eps

    # variance
    x1 = torch.randn(n, m.irreps_in1.dim)
    y1 = torch.randn(n, m.irreps_in2.dim)
    z1 = m(x1, y1).var(0)
    assert z1.mean().log10().abs() < torch.tensor(tol).log10()

    # equivariance
    assert_equivariant(m, irreps_in=[m.irreps_in1, m.irreps_in2], irreps_out=m.irreps_out)
