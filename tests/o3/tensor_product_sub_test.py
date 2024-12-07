from e3nn_jax import Irrep
import torch

from e3nn import o3
from e3nn.nn import Identity
from e3nn.o3 import FullyConnectedTensorProduct, FullTensorProduct, Norm, TensorSquare, ChannelWiseTensorProduct
from e3nn.util.test import assert_equivariant, assert_auto_jitable
from e3nn.util.jit import prepare


def test_fully_connected() -> None:
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")

    def build_module(irreps_in1, irreps_in2, irreps_out):
        return FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

    m = build_module(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))

    assert_equivariant(m)
    assert_auto_jitable(m)
    m_pt2 = torch.compile(prepare(build_module)(irreps_in1, irreps_in2, irreps_out), fullgraph=True)
    m_pt2(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))


def test_fully_connected_normalization() -> None:
    m = FullyConnectedTensorProduct("10x0e", "10x0e", "0e")
    for p in m.parameters():
        p.data.fill_(1.0)

    n = FullyConnectedTensorProduct("3x0e + 7x0e", "3x0e + 7x0e", "0e")
    for p in n.parameters():
        p.data.fill_(1.0)

    x1, x2 = torch.randn(2, 3, 10)
    assert torch.allclose(m(x1, x2), n(x1, x2))


def test_id() -> None:
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")

    def build_module(irreps_in, irreps_out):
        return Identity(irreps_in, irreps_out)

    m = build_module(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))

    assert_equivariant(m)
    assert_auto_jitable(m, strict_shapes=False)

    m_pt2 = torch.compile(prepare(build_module)(irreps_in, irreps_out), fullgraph=True)
    m_pt2(torch.randn(irreps_in.dim))


def test_full() -> None:
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2x2e + 2x3o")

    def build_module(irreps_in1, irreps_in2):
        return FullTensorProduct(irreps_in1, irreps_in2)

    m = build_module(irreps_in1, irreps_in2)
    print(m)

    assert_equivariant(m)
    assert_auto_jitable(m)

    m_pt2 = prepare(build_module)(irreps_in1, irreps_in2)
    m_pt2(irreps_in1.randn(-1), irreps_in2.randn(-1))


def test_channelwise() -> None:
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")

    def build_module(irreps_in1, irreps_in2, irreps_out):
        return ChannelWiseTensorProduct(irreps_in1, irreps_in2, irreps_out)

    m = build_module(irreps_in1, irreps_in2, irreps_out)
    print(m)

    assert_equivariant(m)
    assert_auto_jitable(m)

    m_pt2 = prepare(build_module)(irreps_in1, irreps_in2, irreps_out)
    m_pt2(irreps_in1.randn(-1), irreps_in2.randn(-1))


def test_norm() -> None:
    irreps_in = o3.Irreps("3x0e + 5x1o")
    scalars = torch.randn(3)
    vecs = torch.randn(5, 3)

    def build_module(irreps_in):
        return Norm(irreps_in=irreps_in)

    norm = build_module(irreps_in)
    out_norms = norm(torch.cat((scalars.reshape(1, -1), vecs.reshape(1, -1)), dim=-1))
    true_scalar_norms = torch.abs(scalars)
    true_vec_norms = torch.linalg.norm(vecs, dim=-1)
    assert torch.allclose(out_norms[0, :3], true_scalar_norms)
    assert torch.allclose(out_norms[0, 3:], true_vec_norms)

    assert_equivariant(norm)
    assert_auto_jitable(norm)

    norm_pt2 = torch.compile(prepare(build_module)(irreps_in), fullgraph=True)
    norm_pt2(torch.cat((scalars.reshape(1, -1), vecs.reshape(1, -1)), dim=-1))


def test_square_normalization() -> None:
    irreps = o3.Irreps("0e + 1e + 2e")
    tp = TensorSquare(irreps, irrep_normalization="norm")
    x = irreps.randn(1_000_000, -1, normalization="norm")
    y = tp(x)
    n = Norm(tp.irreps_out, squared=True)(y)

    assert (n.mean(0).log().abs().exp() < 1.1).all()

    irreps = o3.Irreps("0e + 3x1e + 3e")
    tp = o3.TensorSquare(irreps, irrep_normalization="component")
    x = irreps.randn(1_000_000, -1, normalization="component")
    y = tp(x)

    assert (y.pow(2).mean(0).log().abs().exp() < 1.1).all()

    tp = TensorSquare(irreps, irrep_normalization="none")
    y = tp(x)

    assert not (y.pow(2).mean(0).log().abs().exp() < 1.1).all()

    # with weights
    tp = TensorSquare(irreps, irreps)

    n = 2_000
    y = torch.stack([tp(tp.irreps_in.randn(n, -1), torch.randn(tp.weight_numel)) for _ in range(n)])

    assert (y.pow(2).mean([0, 1]).log().abs().exp() < 1.1).all()


def test_square_elasticity_tensor() -> None:
    tp = TensorSquare("1o")
    tp = TensorSquare(tp.irreps_out)
    assert tp.irreps_out.simplify() == o3.Irreps("2x0e + 2x2e + 4e")
