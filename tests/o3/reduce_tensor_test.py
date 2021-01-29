import torch

from e3nn import o3


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance, assert_equivariant):
    tp = o3.ReducedTensorProducts('ijk=-ikj=-jik', i='1e')
    irreps = tp.irreps_out
    Q = tp.change_of_basis

    assert irreps == ((1, (0, 1)),)
    r = o3.rand_angles()
    D = o3.wigner_D(1, *r)
    Q = Q.reshape(3, 3, 3)
    Q1 = torch.einsum('li,mj,nk,ijk', D, D, D, Q)
    assert (Q1 - Q).abs().max() < 10*float_tolerance

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)


def test_reduce_tensor_antisymmetric_L2(float_tolerance, assert_equivariant):
    tp = o3.ReducedTensorProducts('ijk=-ikj=-jik', i='2e')
    irreps = tp.irreps_out
    Q = tp.change_of_basis
    assert irreps[0] == (1, (1, 1))
    q = Q[:3].reshape(3, 5, 5, 5)

    r = o3.rand_angles()
    D1 = o3.wigner_D(1, *r)
    D2 = o3.wigner_D(2, *r)
    Q1 = torch.einsum('il,jm,kn,zijk->zlmn', D2, D2, D2, q)
    Q2 = torch.einsum('yz,zijk->yijk', D1, q)

    assert (Q1 - Q2).abs().max() < 10*float_tolerance

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)


def test_reduce_tensor_elasticity_tensor(assert_equivariant):
    tp = o3.ReducedTensorProducts('ijkl=jikl=klij', i='1e')
    irreps = tp.irreps_out
    assert irreps.dim == 21
    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)


def test_reduce_tensor_elasticity_tensor_parity(assert_equivariant):
    tp = o3.ReducedTensorProducts('ijkl=jikl=klij', i='1o')
    irreps = tp.irreps_out
    assert all(p == 1 for _, (_, p) in irreps)
    assert irreps.dim == 21
    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)


# def test_reduce_tensor_rot():
#     tp = o3.ReducedTensorProducts('ijkl=jikl=klij', i=o3.quaternion_to_matrix, has_parity=False)
#     irreps = tp.irreps_out
#     assert irreps.dim == 21


def test_reduce_tensor_equivariance(float_tolerance, assert_equivariant):
    ir = o3.Irreps('1e')
    tp = o3.ReducedTensorProducts('ijkl=jikl=klij', i=ir)
    irreps = tp.irreps_out
    Q = tp.change_of_basis

    abc = o3.rand_angles()
    R = ir.D_from_angles(*abc)
    D = irreps.D_from_angles(*abc)

    q1 = torch.einsum('qmnop,mi,nj,ok,pl->qijkl', Q, R, R, R, R)
    q2 = torch.einsum('qa,aijkl->qijkl', D, Q)

    assert (q1 - q2).abs().max() < 10*float_tolerance

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
