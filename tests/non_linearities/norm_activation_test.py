# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch

from e3nn import o3, rs
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.non_linearities.rescaled_act import swish


@pytest.mark.parametrize('Rs,normalization,dtype', itertools.product([[2, 3], [(1, 1), (2, 0), (1, 3)]], ['component', 'norm'], [torch.float32, torch.float64]))
def test_norm_activation(Rs, normalization, dtype):
    with o3.torch_default_dtype(dtype):
        m = NormActivation(Rs, swish, normalization=normalization)

        D = rs.rep(Rs, *o3.rand_angles())
        x = rs.randn(2, Rs, normalization=normalization)

        y1 = m(x)
        y1 = torch.einsum('ij,zj->zi', D, y1)

        x2 = torch.einsum('ij,zj->zi', D, x)
        y2 = m(x2)

        assert (y1 - y2).abs().max() < {torch.float32: 1e-5, torch.float64: 1e-10}[dtype]
