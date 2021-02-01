import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.util.test import assert_equivariant


def test_normalization(float_tolerance):
    sqrt_float_tolerance = torch.sqrt(float_tolerance)

    batch, n = 20, 20
    irreps = o3.Irreps("3x0e + 4x1e")

    m = BatchNorm(irreps, normalization='norm')

    x = torch.randn(batch, n, irreps.dim).mul(5.0).add(10.0)
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).sum(3).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    m = BatchNorm(irreps, normalization='component')

    x = torch.randn(batch, n, irreps.dim).mul(5.0).add(10.0)
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).mean(3).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    assert_equivariant(m, irreps_in=irreps, irreps_out=irreps)
