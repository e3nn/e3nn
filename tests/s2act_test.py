# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch
from e3nn import o3
from e3nn.nn import S2Activation


@pytest.mark.parametrize('act, normalization, p_val, p_arg', itertools.product([torch.tanh, lambda x: x**2], ['norm', 'component'], [-1, 1], [-1, 1]))
def test_equivariance(act, normalization, p_val, p_arg):
    torch.set_default_dtype(torch.float64)

    irreps = o3.Irreps.s2signal(3, p_val, p_arg)

    x = irreps.randn(50, -1)
    m = S2Activation(irreps, act, 120, normalization=normalization, lmax_out=6, random_rot=True)

    a, b, c = o3.rand_angles()

    y1 = m(x) @ m.irreps_out.D_from_angles(a, b, c, torch.tensor(1)).T
    y2 = m(x @ m.irreps_in.D_from_angles(a, b, c, torch.tensor(1)).T)

    assert (y1 - y2).abs().max() < 1e-6
