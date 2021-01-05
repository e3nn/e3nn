import torch

from e3nn import o3


def test_wigner_3j_symmetry():
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(1, 3, 2).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 1, 3).transpose(0, 1))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 2, 1).transpose(0, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 1, 2).transpose(0, 1).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 3, 1).transpose(0, 2).transpose(1, 2))


def test_wigner_3j():
    torch.set_default_dtype(torch.float64)
    abc = o3.rand_angles(10)

    l1, l2, l3 = 1, 2, 3
    C = o3.wigner_3j(l1, l2, l3)
    D1 = o3.Irrep(l1, 1).D_from_angles(*abc)
    D2 = o3.Irrep(l2, 1).D_from_angles(*abc)
    D3 = o3.Irrep(l3, 1).D_from_angles(*abc)

    C2 = torch.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    assert (C - C2).abs().max() < 1e-10
