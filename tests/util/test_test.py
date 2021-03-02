import pytest

import torch

from e3nn.util.test import assert_equivariant, assert_auto_jitable


@pytest.mark.xfail
def test_assert_equivariant():
    def not_equivariant(x1, x2):
        return x1*x2
    not_equivariant.irreps_in1 = "2x0e + 1x1e + 3x2o + 1x4e"
    not_equivariant.irreps_in2 = "2x0o + 3x0o + 3x2e + 1x4o"
    assert_equivariant(not_equivariant)


@pytest.mark.xfail
def test_jit_trace():
    def not_tracable(param):
        if param.shape[0] == 7:
            return torch.ones(8)
        else:
            return torch.randn(8, 3)
    not_tracable.irreps_in = "2x0e"
    not_tracable.irreps_out = "1x1o"
    assert_auto_jitable(not_tracable)
