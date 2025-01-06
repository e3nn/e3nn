import torch

import functools

import pytest
from e3nn.nn import FullyConnectedNet
from e3nn.util.test import assert_auto_jitable, assert_torch_compile


@pytest.mark.parametrize("act", [None, torch.tanh])
@pytest.mark.parametrize("var_in, var_out, out_act", [(1, 1, False), (1, 1, True), (0.1, 10.0, False), (0.1, 0.05, True)])
def test_variance(act, var_in, var_out, out_act) -> None:
    hs = (1000, 500, 1500, 4)

    f = FullyConnectedNet(hs, act, var_in, var_out, out_act)
    x = torch.randn(2000, hs[0]) * var_in**0.5

    y = f(x) / var_out**0.5

    if not out_act:
        assert y.mean().abs() < 0.5
    assert y.pow(2).mean().log10().abs() < torch.tensor(1.5).log10()

    f = assert_auto_jitable(f)
    f(x)

    f_pt2 = assert_torch_compile(
        'inductor',
        functools.partial(FullyConnectedNet, hs, act, var_in, var_out, out_act),
        x
    )
    f_pt2(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_data_parallel() -> None:
    fc = torch.nn.DataParallel(FullyConnectedNet([10, 20, 30]).cuda())
    y = fc(torch.randn(32, 10).cuda())
    y.sum().backward()
