import torch

from e3nn import o3


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance):
    irreps, Q = o3.reduce_tensor('ijk=-ikj=-jik', i='1e')
    assert irreps == ((1, (0, 1)),)
    r = o3.rand_angles()
    D = o3.wigner_D(1, *r)
    Q = Q.reshape(3, 3, 3)
    Q1 = torch.einsum('li,mj,nk,ijk', D, D, D, Q)
    assert (Q1 - Q).abs().max() < 10*float_tolerance


def test_reduce_tensor_antisymmetric_L2(float_tolerance):
    irreps, Q = o3.reduce_tensor('ijk=-ikj=-jik', i='2e')
    assert irreps[0] == (1, (1, 1))
    q = Q[:3].reshape(3, 5, 5, 5)

    r = o3.rand_angles()
    D1 = o3.wigner_D(1, *r)
    D2 = o3.wigner_D(2, *r)
    Q1 = torch.einsum('il,jm,kn,zijk->zlmn', D2, D2, D2, q)
    Q2 = torch.einsum('yz,zijk->yijk', D1, q)

    assert (Q1 - Q2).abs().max() < 10*float_tolerance
    assert (q + q.transpose(1, 2)).abs().max() < 10*float_tolerance
    assert (q + q.transpose(1, 3)).abs().max() < 10*float_tolerance
    assert (q + q.transpose(3, 2)).abs().max() < 10*float_tolerance


def test_reduce_tensor_elasticity_tensor():
    irreps, _Q = o3.reduce_tensor('ijkl=jikl=klij', i='1e')
    assert irreps.dim == 21


def test_reduce_tensor_elasticity_tensor_parity():
    irreps, _Q = o3.reduce_tensor('ijkl=jikl=klij', i='1o')
    assert all(p == 1 for _, (_, p) in irreps)
    assert irreps.dim == 21


def test_reduce_tensor_rot():
    irreps, _Q = o3.reduce_tensor('ijkl=jikl=klij', i=o3.quaternion_to_matrix, has_parity=False)
    assert irreps.dim == 21


def test_reduce_tensor_equivariance(float_tolerance):
    ir = o3.Irreps('1e')
    irreps, Q = o3.reduce_tensor('ijkl=jikl=klij', i=ir)

    abc = o3.rand_angles()
    R = ir.D_from_angles(*abc)
    D = irreps.D_from_angles(*abc)

    q1 = torch.einsum('qmnop,mi,nj,ok,pl->qijkl', Q, R, R, R, R)
    q2 = torch.einsum('qa,aijkl->qijkl', D, Q)

    assert (q1 - q2).abs().max() < 10*float_tolerance
