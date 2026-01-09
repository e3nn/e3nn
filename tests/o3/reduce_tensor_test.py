import tempfile
import functools

import torch

from e3nn import o3
from e3nn.util.test import assert_auto_jitable, assert_equivariant, assert_torch_compile


def test_save_load() -> None:
    tp1 = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(tp1.state_dict(), tmp.name)
        tp2 = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
        tp2.load_state_dict(torch.load(tmp.name, weights_only=False))

    xs = (torch.randn(2, 5 + 3), torch.randn(2, 5 + 3))
    assert torch.allclose(tp1(*xs), tp2(*xs))

    assert torch.allclose(tp1.change_of_basis, tp2.change_of_basis)


def test_antisymmetric_matrix(float_tolerance) -> None:

    tp = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")

    Q = tp.change_of_basis
    x = torch.randn(2, 5 + 3)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_torch_compile("inductor", functools.partial(o3.ReducedTensorProducts, "ij=-ji", i="5x0e + 1e"), *x)
    assert_auto_jitable(tp)

    assert (tp(*x) - torch.einsum("xij,i,j", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xij->xji", Q)).abs().max() < float_tolerance


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="1e")
    assert tp.irreps_out == o3.Irreps("0e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(3, 3)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_antisymmetric_L2(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="2e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(3, 5)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1e")
    assert tp.irreps_out.dim == 21

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor_parity(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1o")
    assert tp.irreps_out.dim == 21
    assert all(ir.p == 1 for _, ir in tp.irreps_out)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance
