import pytest
import torch

from e3nn import o3


def test_wigner_3j_symmetry() -> None:
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(1, 3, 2).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 1, 3).transpose(0, 1))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 2, 1).transpose(0, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 1, 2).transpose(0, 1).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 3, 1).transpose(0, 2).transpose(1, 2))


@pytest.mark.parametrize("l1,l2,l3", [(1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 2, 2)])
def test_wigner_3j(l1, l2, l3, float_tolerance) -> None:
    abc = o3.rand_angles(10)

    C = o3.wigner_3j(l1, l2, l3)
    D1 = o3.Irrep(l1, 1).D_from_angles(*abc)
    D2 = o3.Irrep(l2, 1).D_from_angles(*abc)
    D3 = o3.Irrep(l3, 1).D_from_angles(*abc)

    C2 = torch.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    assert (C - C2).abs().max() < float_tolerance


def test_cartesian(float_tolerance) -> None:
    abc = o3.rand_angles(10)
    R = o3.angles_to_matrix(*abc)
    D = o3.wigner_D(1, *abc)
    assert (R - D).abs().max() < float_tolerance


def commutator(A, B):
    return A @ B - B @ A


@pytest.mark.parametrize("j", [0, 1 / 2, 1, 3 / 2, 2, 5 / 2])
def test_su2_algebra(j, float_tolerance) -> None:
    X = o3.su2_generators(j)
    assert torch.allclose(commutator(X[0], X[1]), X[2], atol=float_tolerance)
    assert torch.allclose(commutator(X[1], X[2]), X[0], atol=float_tolerance)
