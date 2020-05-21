# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from e3nn.tensor_product import LearnableTensorSquare, LearnableTensorProduct
from e3nn import rs


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
