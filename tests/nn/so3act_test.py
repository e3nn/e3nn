# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch
from e3nn import o3
from e3nn.nn import SO3Activation
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize('act, normalization, p_val', itertools.product([torch.tanh, lambda x: x**2], ['component'], [-1, 1]))
def test_equivariance(float_tolerance, act, normalization, p_val):
    irreps = o3.Irreps([(2 * l + 1, (l, p_val)) for l in range(5 + 1)])

    m = SO3Activation(irreps, act, 64 * 1024, normalization=normalization)

    assert_equivariant(m, ntrials=10, tolerance=0.4)
