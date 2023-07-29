import pytest

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.test import assert_equivariant, assert_auto_jitable, assert_normalized, random_irreps


def test_assert_equivariant() -> None:
    def not_equivariant(x1, x2):
        return x1 * x2

    not_equivariant.irreps_in1 = o3.Irreps("2x0e + 1x1e + 3x2o + 1x4e")
    not_equivariant.irreps_in2 = o3.Irreps("2x0o + 3x0o + 3x2e + 1x4o")
    not_equivariant.irreps_out = o3.Irreps("1x1e + 2x0o + 3x2e + 1x4o")
    assert not_equivariant.irreps_in1.dim == not_equivariant.irreps_in2.dim
    assert not_equivariant.irreps_in1.dim == not_equivariant.irreps_out.dim
    with pytest.raises(AssertionError):
        assert_equivariant(not_equivariant)


def test_jit_trace() -> None:
    @compile_mode("trace")
    class NotTracable(torch.nn.Module):
        def forward(self, param):
            if param.shape[0] == 7:
                return torch.ones(8)
            else:
                return torch.randn(8, 3)

    not_tracable = NotTracable()
    not_tracable.irreps_in = o3.Irreps("2x0e")
    not_tracable.irreps_out = o3.Irreps("1x1o")
    # TorchScript returns some weird exceptions...
    with pytest.raises(Exception):
        assert_auto_jitable(not_tracable)


def test_bad_normalize() -> None:
    def not_normal(x1) -> float:
        return 870.0 * x1.square().relu()

    not_normal.irreps_in = random_irreps(clean=True, allow_empty=False)
    not_normal.irreps_out = not_normal.irreps_in
    with pytest.raises(AssertionError):
        assert_normalized(not_normal)


def test_normalized_ident() -> None:
    def ident(x1):
        return x1

    ident.irreps_in = random_irreps(clean=True, allow_empty=False)
    ident.irreps_out = ident.irreps_in
    assert_normalized(ident)
