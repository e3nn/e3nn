import torch

from e3nn import o3
from e3nn.nn import Identity
from e3nn.o3 import FullyConnectedTensorProduct, Linear, FullTensorProduct, Norm
from e3nn.util.test import assert_equivariant, assert_jit_trace


def test_fully_connected():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))

    assert_equivariant(m)
    assert_jit_trace(m)


def test_id():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = Identity(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))

    assert_equivariant(m)
    assert_jit_trace(m, strict_shapes=False)


def test_linear():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = Linear(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))

    assert_equivariant(m)
    assert_jit_trace(m)


def test_full():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2x2e + 2x3o")
    m = FullTensorProduct(irreps_in1, irreps_in2)
    print(m)

    assert_equivariant(m)
    assert_jit_trace(m)


def test_norm():
    irreps_in = o3.Irreps("3x0e + 5x1o")
    scalars = torch.randn(3)
    vecs = torch.randn(5, 3)
    norm = Norm(irreps_in=irreps_in)
    out_norms = norm(
        torch.cat((
            scalars.reshape(1, -1), vecs.reshape(1, -1)
        ), dim=-1)
    )
    true_scalar_norms = torch.abs(scalars)
    true_vec_norms = torch.linalg.norm(vecs, dim=-1)
    assert torch.allclose(out_norms[0, :3], true_scalar_norms)
    assert torch.allclose(out_norms[0, 3:], true_vec_norms)

    assert_equivariant(norm)
    assert_jit_trace(norm)
