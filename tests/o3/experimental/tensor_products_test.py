import torch
import numpy as np
import pytest

from e3nn import o3


def test_tensor_product(dtype=torch.float32):
    x1 = o3.experimental.IrrepsArray("1o", torch.Tensor([1.0, 0.0, 0.0]).to(dtype=dtype))
    x2 = o3.experimental.IrrepsArray("1o", torch.Tensor([0.0, 1.0, 0.0]).to(dtype=dtype))
    x3 = torch.compile(o3.experimental.tensor_product, fullgraph=True)(x1, x2, filter_ir_out=("1e",))
    assert x3.irreps == o3.Irreps("1e")
    np.testing.assert_allclose(x3.array, torch.Tensor([0.0, 0.0, 1 / 2**0.5]))


def test_tensor_product_with_zeros():
    x1 = o3.experimental.from_chunks("1o", [None], (), torch.float32)
    x2 = o3.experimental.IrrepsArray("1o", torch.Tensor([0.0, 1.0, 0.0]))
    x3 = o3.experimental.tensor_product(x1, x2)
    assert x3.irreps == o3.Irreps("0e + 1e + 2e")
    assert x3.zero_flags == (True, True, True)


def test_elementwise_with_zeros():
    x1 = o3.experimental.from_chunks("1o", [None], (), torch.float32)
    x2 = o3.experimental.IrrepsArray("1o", torch.Tensor([0.0, 1.0, 0.0]))
    x3 = o3.experimental.elementwise_tensor_product(x1, x2)
    assert x3.irreps == o3.Irreps("0e + 1e + 2e")
    assert x3.zero_flags == (True, True, True)


def test_tensor_product_irreps():
    irreps = o3.experimental.tensor_product("1o", "1o", filter_ir_out=("1e",))
    assert irreps == o3.Irreps("1e")


def test_elementwise_tensor_product():
    x1 = o3.experimental.normal("0e + 1o", (10,))
    x2 = o3.experimental.normal("1o + 0o", (20, 1))

    x3 = o3.experimental.elementwise_tensor_product(x1, x2)
    assert x3.irreps == o3.Irreps("1o + 1e")
    assert x3.shape == (20, 10, 6)


def test_square_normalization_1():
    x = o3.experimental.normal("1o", (100_000,))
    y = o3.experimental.tensor_square(x)
    np.testing.assert_array_less(torch.exp(torch.abs(torch.log(torch.mean(y.array**2, 0)))), 1.1)


def test_square_normalization_2():
    x = o3.experimental.normal("2e + 1o", (100_000,))
    y = o3.experimental.tensor_square(x)
    np.testing.assert_array_less(torch.exp(torch.abs(torch.log(torch.mean(y.array**2, 0)))), 1.1)


# def test_square_normalization_3():
#     x = o3.experimental.normal("2e + 5x1o", (100_000,))
#     y = o3.experimental.tensor_square(x)
#     torch.testing.utils.assert_array_less(
#         torch.exp(torch.abs(torch.log(torch.mean(y.array**2, 0)))), 1.1
#     )


def test_tensor_square_normalization():
    x = o3.experimental.normal("2x0e + 2x0o + 1o + 1e", (10_000,))
    y = o3.experimental.tensor_square(x, irrep_normalization="component")
    torch.testing.assert_allclose(
        o3.experimental.mean(o3.experimental.norm(y, squared=True), axis=0).array,
        torch.Tensor([ir.dim for mul, ir in y.irreps for _ in range(mul)]),
        atol=0.1,
        rtol=0.1,
    )

    x = o3.experimental.normal("2x0e + 2x0o + 1o + 1e", (10_000,), normalize=True)
    y = o3.experimental.tensor_square(x, normalized_input=True, irrep_normalization="norm")
    torch.testing.assert_allclose(
        o3.experimental.mean(o3.experimental.norm(y, squared=True), axis=0).array, 1.0, atol=0.1, rtol=0.1
    )

    x = o3.experimental.normal("2x0e + 2x0o + 1o + 1e", (10_000,), normalize=True)
    y = o3.experimental.tensor_square(x, normalized_input=True, irrep_normalization="component")
    torch.testing.assert_allclose(
        o3.experimental.mean(o3.experimental.norm(y, squared=True), axis=0).array,
        torch.Tensor([ir.dim for mul, ir in y.irreps for _ in range(mul)]),
        rtol=0.1,
    )

    x = o3.experimental.normal("2x0e + 2x0o + 1o + 1e", (10_000,), normalization="norm")
    y = o3.experimental.tensor_square(x, irrep_normalization="norm")
    torch.testing.assert_allclose(o3.experimental.mean(o3.experimental.norm(y, squared=True), axis=0).array, 1.0, rtol=0.1)


# def test_tensor_square_and_spherical_harmonics(keys):
#     x = e3nn.normal("1o", keys[0])

#     y1 = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="norm")["2e"]
#     y2 = e3nn.spherical_harmonics("2e", x, normalize=False, normalization="norm")
#     np.testing.assert_allclose(y1.array, y2.array, atol=1e-6)

#     y1 = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="component")[
#         "2e"
#     ]
#     y2 = e3nn.spherical_harmonics("2e", x, normalize=False, normalization="component")
#     np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)

#     # normalize the input
#     y1 = e3nn.tensor_square(
#         x / e3nn.norm(x), normalized_input=True, irrep_normalization="component"
#     )["2e"]
#     y2 = e3nn.spherical_harmonics("2e", x, normalize=True, normalization="component")
#     np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)


# def test_tensor_square_equivariant(keys):
#     e3nn.utils.assert_equivariant(
#         e3nn.tensor_square, keys[0], "2x0e + 2x1o + 2x1e + 2x2e"
#     )


# def test_tensor_square_dtype():
#     jax.config.update("jax_enable_x64", True)
#     x = e3nn.IrrepsArray("1o", jnp.array([1.0, 0.0, 0.0]))
#     e3nn.utils.assert_output_dtype_matches_input_dtype(e3nn.tensor_square, x)


# def test_tensor_product_dtype():
#     jax.config.update("jax_enable_x64", True)
#     x1 = e3nn.IrrepsArray("1o", jnp.array([1.0, 0.0, 0.0]))
#     x2 = e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0]))
#     e3nn.utils.assert_output_dtype_matches_input_dtype(e3nn.tensor_product, x1, x2)


# def test_elementwise_dtype():
#     torch.set_default_dtype(torch.float64)
#     x1 = o3.experimental.IrrepsArray("1o", torch.Tensor([1.0, 0.0, 0.0]))
#     x2 = o3.experimental.IrrepsArray("1o", torch.Tensor([0.0, 1.0, 0.0]))
#     utils.assert_output_dtype_matches_input_dtype(
#         o3.experimental.elementwise_tensor_product, x1, x2
#     )
