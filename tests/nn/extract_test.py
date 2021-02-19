import torch

from e3nn.nn import Extract, ExtractIr
from e3nn.util.test import assert_auto_jitable


def test_extract():
    c = Extract('1e + 0e + 0e', ['0e', '0e'], [(1,), (2,)])
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert out == (torch.Tensor([1.]), torch.Tensor([2.]))
    assert_auto_jitable(c)


def test_extract_ir():
    c = ExtractIr('1e + 0e + 0e', '0e')
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert torch.all(out == torch.Tensor([1., 2.]))
    assert_auto_jitable(c)
