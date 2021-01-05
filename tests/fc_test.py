import torch

import pytest
from e3nn.nn import FullyConnectedNet


@pytest.mark.parametrize('var_in, var_out, out_act', [(1, 1, False), (1, 1, True), (0.1, 10.0, False), (0.1, 0.05, True)])
def test_variance(var_in, var_out, out_act):
    hs = (256, 128, 256, 8)
    act = None

    f = FullyConnectedNet(hs, act, var_in, var_out, out_act)

    x = torch.randn(1_000, hs[0]) * var_in**0.5
    y = f(x) / var_out**0.5

    assert y.var(0).mean().log10().abs() < torch.tensor(1.5).log10()
