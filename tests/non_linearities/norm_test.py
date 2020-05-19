# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest

from e3nn import rs
from e3nn.non_linearities.norm import Norm


@pytest.mark.parametrize('Rs,normalization', itertools.product([[2, 3], [(1, 1), (2, 0), (1, 3)]], ['component', 'norm']))
def test_norm(Rs, normalization):
    m = Norm(Rs, normalization=normalization)
    x = rs.randn(2, Rs, normalization=normalization)
    x = m(x)
    assert x.shape == (2, rs.mul_dim(Rs))
