# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch
from e3nn import o3
from e3nn.nn import SO3Activation
from e3nn.util.test import assert_equivariant
from e3nn.util.jit import compile


@pytest.mark.parametrize('act, normalization, p_val', itertools.product([torch.tanh, lambda x: x**2], ['component'], [-1, 1]))
def test_equivariance(act, normalization, p_val):
    irreps = o3.Irreps([(2 * l + 1, (l, p_val)) for l in range(5 + 1)])

    m = SO3Activation(irreps, act, 6, normalization=normalization)

    assert_equivariant(m, ntrials=10, tolerance=0.04)


@pytest.mark.parametrize('aspect_ratio', [1, 2, 3, 4])
def test_identity(aspect_ratio):
    irreps = o3.Irreps([(2 * l + 1, (l, 1)) for l in range(5 + 1)])

    m = SO3Activation(irreps, lambda x: x, 6, aspect_ratio=aspect_ratio)
    m = compile(m)

    x = irreps.randn(-1)
    y = m(x)

    mse = (x - y).pow(2).mean()
    assert mse < 1e-5, mse
