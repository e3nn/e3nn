import pytest

import copy

import torch

from e3nn.nn import Extract, ExtractIr
from e3nn.util.test import assert_auto_jitable, assert_equivariant
from e3nn.util.jit import prepare


def test_extract() -> None:

    def build_module():
        return Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])

    c = build_module()
    c_pt2 = torch.compile(prepare(build_module)(), fullgraph=True)
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    out_pt2 = c_pt2(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert out == (torch.Tensor([1.0]), torch.Tensor([2.0]))
    assert out == out_pt2
    assert_auto_jitable(c)
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


@pytest.mark.parametrize("squeeze", [True, False])
def test_extract_single(squeeze) -> None:

    def build_module():
        return Extract("1e + 0e + 0e", ["0e"], [(1,)], squeeze_out=squeeze)

    c = build_module()
    c_pt2 = torch.compile(prepare(build_module)(), fullgraph=True)
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    out_pt2 = c_pt2(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    if squeeze:
        assert isinstance(out, torch.Tensor)
    else:
        assert len(out) == 1
        out = out[0]
        assert len(out_pt2) == 1
        out_pt2 = out_pt2[0]
    assert out == torch.Tensor([1.0])
    assert out_pt2 == torch.Tensor([1.0])
    assert_auto_jitable(c)
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


def test_extract_ir() -> None:
    build_module = lambda: ExtractIr("1e + 0e + 0e", "0e")
    c = build_module()
    c_pt2 = torch.compile(prepare(build_module)(), fullgraph=True)
    out = c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    out_pt2 = c_pt2(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert torch.all(out == torch.Tensor([1.0, 2.0]))
    assert torch.all(out_pt2 == out)
    assert_auto_jitable(c)
    assert_equivariant(c)


def test_copy() -> None:
    c = Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])
    _ = copy.deepcopy(c)
    c = ExtractIr("1e + 0e + 0e", "0e")
    _ = copy.deepcopy(c)
