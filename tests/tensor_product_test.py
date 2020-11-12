# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn import o3, rs
from e3nn.tensor_product import (CustomWeightedTensorProduct,
                                 LearnableTensorProduct, LearnableTensorSquare,
                                 WeightedTensorProduct)


def test_learnable_tensor_square_normalization():
    Rs_in = [1, 2, 3, 4]
    Rs_out = [0, 2, 4, 5]

    m = LearnableTensorSquare(Rs_in, Rs_out)
    y = m(rs.randn(1000, Rs_in))

    assert y.var().log10().abs() < 1.5, y.var().item()


def test_learnable_tensor_product_normalization():
    Rs_in1 = [2, 0, 4]
    Rs_in2 = [2, 3]
    Rs_out = [0, 2, 4, 5]

    m = LearnableTensorProduct(Rs_in1, Rs_in2, Rs_out)

    x1 = rs.randn(1000, Rs_in1)
    x2 = rs.randn(1000, Rs_in2)
    y = m(x1, x2)

    assert y.var().log10().abs() < 1.5, y.var().item()


def test_weighted_tensor_product():
    torch.set_default_dtype(torch.float64)

    Rs_in1 = rs.simplify([1] * 20 + [2] * 4)
    Rs_in2 = rs.simplify([0] * 10 + [1] * 10 + [2] * 5)
    Rs_out = rs.simplify([0] * 3 + [1] * 4)

    tp = WeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, groups=2)

    x1 = rs.randn(20, Rs_in1)
    x2 = rs.randn(20, Rs_in2)

    angles = o3.rand_angles()

    z1 = tp(x1, x2) @ rs.rep(Rs_out, *angles).T
    z2 = tp(x1 @ rs.rep(Rs_in1, *angles).T, x2 @ rs.rep(Rs_in2, *angles).T)

    z1.sum().backward()

    assert torch.allclose(z1, z2)


def test_custom_weighted_tensor_product():
    torch.set_default_dtype(torch.float64)

    Rs_in1 = [(20, 1), (4, 2)]
    Rs_in2 = [(10, 0), (10, 1), (4, 2)]
    Rs_out = [(3, 0), (4, 1)]

    instr = [
        (0, 1, 0, 'uvw'),
        (1, 2, 1, 'uuu'),
        (0, 1, 1, 'uvw'),
    ]

    tp = CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr)

    x1 = rs.randn(20, Rs_in1)
    x2 = rs.randn(20, Rs_in2)

    angles = o3.rand_angles()

    z1 = tp(x1, x2) @ rs.rep(Rs_out, *angles).T
    z2 = tp(x1 @ rs.rep(Rs_in1, *angles).T, x2 @ rs.rep(Rs_in2, *angles).T)

    z1.sum().backward()

    assert torch.allclose(z1, z2)
