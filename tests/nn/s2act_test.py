import itertools

import pytest
import torch
from e3nn import io
from e3nn.nn import S2Activation
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize(
    "act, normalization, p_val, p_arg",
    itertools.product([torch.tanh, lambda x: x**2], ["norm", "component"], [-1, 1], [-1, 1]),
)
def test_equivariance(float_tolerance, act, normalization, p_val, p_arg) -> None:
    irreps = io.SphericalTensor(3, p_val, p_arg)

    m = S2Activation(irreps, act, 120, normalization=normalization, lmax_out=6, random_rot=True)

    assert_equivariant(m, ntrials=10, tolerance=torch.sqrt(float_tolerance))
