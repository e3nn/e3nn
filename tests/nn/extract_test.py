import pytest

import copy

import torch

from e3nn.nn import Extract, ExtractIr
from e3nn.util.test import assert_auto_jitable, assert_equivariant


def test_extract() -> None:
    c = Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert out == (torch.Tensor([1.0]), torch.Tensor([2.0]))
    assert_auto_jitable(c)
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


@pytest.mark.parametrize("squeeze", [True, False])
def test_extract_single(squeeze) -> None:
    c = Extract("1e + 0e + 0e", ["0e"], [(1,)], squeeze_out=squeeze)
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    if squeeze:
        assert isinstance(out, torch.Tensor)
    else:
        assert len(out) == 1
        out = out[0]
    assert out == torch.Tensor([1.0])
    assert_auto_jitable(c)
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


def test_extract_ir() -> None:
    c = ExtractIr("1e + 0e + 0e", "0e")
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert torch.all(out == torch.Tensor([1.0, 2.0]))
    assert_auto_jitable(c)
    assert_equivariant(c)


def test_copy() -> None:
    c = Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])
    _ = copy.deepcopy(c)
    c = ExtractIr("1e + 0e + 0e", "0e")
    _ = copy.deepcopy(c)
