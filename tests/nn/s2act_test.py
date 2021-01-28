# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch
from e3nn import o3, io
from e3nn.nn import S2Activation


@pytest.mark.parametrize('act, normalization, p_val, p_arg', itertools.product([torch.tanh, lambda x: x**2], ['norm', 'component'], [-1, 1], [-1, 1]))
def test_equivariance(assert_equivariant, act, normalization, p_val, p_arg):
    irreps = io.SphericalTensor(3, p_val, p_arg)

    m = S2Activation(irreps, act, 120, normalization=normalization, lmax_out=6, random_rot=True)

    assert_equivariant(m, ntrials=10, sqrt_tolerance=True)
