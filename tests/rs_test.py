# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, protected-access
from functools import partial

import pytest
import torch

from e3nn import o3, rs
from e3nn.non_linearities.norm import Norm


def test_convention():
    Rs = [0]
    assert rs.convention(Rs) == [(1, 0, 0)]
    Rs = [0, (2, 0)]
    assert rs.convention(Rs) == [(1, 0, 0), (2, 0, 0)]


def test_simplify():
    Rs = [(1, 0), 0, (1, 0)]
    assert rs.simplify(Rs) == [(3, 0, 0)]


def test_irrep_dim():
    Rs = [(1, 0), (3, 1), (2, 2)]
    assert rs.irrep_dim(Rs) == 1 + 3 + 5
    Rs = [(1, 0), (3, 0), (2, 0)]
    assert rs.irrep_dim(Rs) == 1 + 1 + 1


def test_mul_dim():
    Rs = [(1, 0), (3, 1), (2, 2)]
    assert rs.mul_dim(Rs) == 6
    Rs = [(1, 0), (3, 0), (2, 0)]
    assert rs.mul_dim(Rs) == 6


def test_dim():
    Rs = [(1, 0), (3, 1), (2, 2)]
    assert rs.dim(Rs) == 1 * 1 + 3 * 3 + 2 * 5
    Rs = [(1, 0), (3, 0), (2, 0)]
    assert rs.dim(Rs) == 1 * 1 + 3 * 1 + 2 * 1


def test_map_irrep_to_Rs():
    with o3.torch_default_dtype(torch.float64):
        Rs = [(3, 0)]
        mapping_matrix = rs.map_irrep_to_Rs(Rs)
        assert torch.allclose(mapping_matrix, torch.ones(3, 1))

        Rs = [(1, 0), (1, 1), (1, 2)]
        mapping_matrix = rs.map_irrep_to_Rs(Rs)
        assert torch.allclose(mapping_matrix, torch.eye(1 + 3 + 5))


def test_map_mul_to_Rs():
    with o3.torch_default_dtype(torch.float64):
        Rs = [(3, 0)]
        mapping_matrix = rs.map_mul_to_Rs(Rs)
        assert torch.allclose(mapping_matrix, torch.eye(3))

        Rs = [(1, 0), (1, 1), (1, 2)]
        mapping_matrix = rs.map_mul_to_Rs(Rs)
        check_matrix = torch.zeros(1 + 3 + 5, 3)
        check_matrix[0, 0] = 1.
        check_matrix[1:4, 1] = 1.
        check_matrix[4:, 2] = 1.
        assert torch.allclose(mapping_matrix, check_matrix)


############################################################################


def test_elementwise_tensor_product():
    torch.set_default_dtype(torch.float64)

    Rs_1 = [(3, 0), (2, 1), (5, 2)]
    Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

    Rs_out, m = rs.elementwise_tensor_product(Rs_1, Rs_2)
    mul = rs.ElementwiseTensorProduct(Rs_1, Rs_2)

    x1 = torch.randn(1, rs.dim(Rs_1))
    x2 = torch.randn(1, rs.dim(Rs_2))

    y1 = mul(x1, x2)
    y2 = torch.einsum('zi,zj->ijz', x1, x2)
    y2 = m @ y2.reshape(-1, y2.shape[2])
    y2 = y2.T

    assert rs.dim(Rs_out) == y1.shape[1]
    assert (y1 - y2).abs().max() < 1e-10


############################################################################


def test_tensor_square_equivariance():
    with o3.torch_default_dtype(torch.float64):
        Rs_in = [(3, 0), (2, 1), (5, 2)]

        sq = rs.TensorSquare(Rs_in, o3.selection_rule)

        x = rs.randn(Rs_in)

        abc = o3.rand_angles()
        D_in = rs.rep(Rs_in, *abc)
        D_out = rs.rep(sq.Rs_out, *abc)

        y1 = sq(D_in @ x)
        y2 = D_out @ sq(x)

        assert (y1 - y2).abs().max() < 1e-10


def test_tensor_square_norm():
    for Rs_in in [[(1, 0), (1, 1)]]:
        with o3.torch_default_dtype(torch.float64):
            Rs_out, Q = rs.tensor_square(Rs_in, o3.selection_rule, normalization='component', sorted=True)

            I1 = (Q @ Q.t()).to_dense()
            I2 = torch.eye(rs.dim(Rs_out))

            d = (I1 - I2).pow(2).mean().sqrt()
            assert d < 1e-10


############################################################################


def test_format():
    assert rs.format_Rs([]) == ""
    assert rs.format_Rs([2]) == "2"


############################################################################


def test_tensor_product_equal_TensorProduct():
    with o3.torch_default_dtype(torch.float64):
        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        Rs_out, m = rs.tensor_product(Rs_1, Rs_2, o3.selection_rule, sorted=True)
        mul = rs.TensorProduct(Rs_1, Rs_2, o3.selection_rule)

        x1 = rs.randn(1, Rs_1)
        x2 = rs.randn(1, Rs_2)

        y1 = mul(x1, x2)
        y2 = torch.einsum('zi,zj->ijz', x1, x2)
        y2 = (m @ y2.reshape(rs.dim(Rs_1) * rs.dim(Rs_2), -1)).T

        assert rs.dim(Rs_out) == y1.shape[1]
        assert (y1 - y2).abs().max() < 1e-10 * y1.abs().max()


def test_tensor_product_to_dense():
    with o3.torch_default_dtype(torch.float64):
        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        mul = rs.TensorProduct(Rs_1, Rs_2, o3.selection_rule)
        assert mul.to_dense().shape == (rs.dim(mul.Rs_out), rs.dim(Rs_1), rs.dim(Rs_2))


def test_tensor_product_symmetry():
    with o3.torch_default_dtype(torch.float64):
        Rs_in = [(3, 0), (2, 1), (5, 2)]
        Rs_out = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        mul1 = rs.TensorProduct(Rs_in, o3.selection_rule, Rs_out)
        mul2 = rs.TensorProduct(o3.selection_rule, Rs_in, Rs_out)

        assert mul1.Rs_in2 == mul2.Rs_in1

        x = torch.randn(rs.dim(Rs_in), rs.dim(mul1.Rs_in2))
        y1 = mul1(x)
        y2 = mul2(x.T)

        assert (y1 - y2).abs().max() < 1e-10


def test_tensor_product_left_right():
    with o3.torch_default_dtype(torch.float64):
        Rs_1 = [(3, 0), (2, 1), (5, 2)]
        Rs_2 = [(1, 0), (2, 1), (2, 2), (2, 0), (2, 1), (1, 2)]

        mul = rs.TensorProduct(Rs_1, Rs_2, o3.selection_rule)

        x1 = rs.randn(2, Rs_1)
        x2 = rs.randn(2, Rs_2)

        y0 = mul(x1, x2)

        y1 = mul(torch.einsum('zi,zj->zij', x1, x2))
        assert (y0 - y1).abs().max() < 1e-10 * y0.abs().max()

        mul._complete = 'in1'
        y1 = mul(x1, x2)
        assert (y0 - y1).abs().max() < 1e-10 * y0.abs().max()

        mul._complete = 'in2'
        y1 = mul(x1, x2)
        assert (y0 - y1).abs().max() < 1e-10 * y0.abs().max()


@pytest.mark.parametrize('Rs_in1, Rs_in2', [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(2, 0), (1, 1), (1, 3)])])
def test_tensor_product_in_in_normalization(Rs_in1, Rs_in2):
    with o3.torch_default_dtype(torch.float64):
        Rs_out, Q = rs.tensor_product(Rs_in1, Rs_in2, o3.selection_rule)

        n = rs.dim(Rs_out)
        I = torch.eye(n)

        d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10

        d = ((Q.t() @ Q).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10


@pytest.mark.parametrize('Rs_in1, Rs_in2', [([0], [0]), ([4, 2], [3, 4])])
def test_tensor_product_in_in_normalization_norm(Rs_in1, Rs_in2):
    with o3.torch_default_dtype(torch.float64):
        tp = rs.TensorProduct(Rs_in1, Rs_in2, o3.selection_rule, normalization='norm')

        x1 = rs.randn(10, Rs_in1, normalization='norm')
        x2 = rs.randn(10, Rs_in2, normalization='norm')

        n = Norm(tp.Rs_out, normalization='norm')
        x = n(tp(x1, x2)).mean(0)
        assert (x.log10().abs() < 1).all()


@pytest.mark.parametrize('Rs_in1, Rs_out', [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(2, 0), (1, 1), (1, 3)])])
def test_tensor_product_in_out_normalization(Rs_in1, Rs_out):
    with o3.torch_default_dtype(torch.float64):
        n = rs.dim(Rs_out)
        I = torch.eye(n)

        _, Q = rs.tensor_product(Rs_in1, o3.selection_rule, Rs_out)
        d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10

        _, Q = rs.tensor_product(o3.selection_rule, Rs_in1, Rs_out)
        d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10


@pytest.mark.parametrize('Rs_in1, Rs_out', [([(1, 0)], [(2, 0)]), ([(3, 1), (2, 2)], [(1, 1), (2, 2)])])
def test_tensor_product_in_out_normalization_l0(Rs_in1, Rs_out):
    with o3.torch_default_dtype(torch.float64):
        n = rs.dim(Rs_out)
        I = torch.eye(n)

        _, Q = rs.tensor_product(Rs_in1, partial(o3.selection_rule, lmax=0), Rs_out)
        d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10

        _, Q = rs.tensor_product(partial(o3.selection_rule, lmax=0), Rs_in1, Rs_out)
        d = ((Q @ Q.t()).to_dense() - I).pow(2).mean().sqrt()
        assert d < 1e-10

############################################################################


def test_reduce_tensor_Levi_Civita_symbol():
    Rs, Q = rs.reduce_tensor('ijk=-ikj=-jik', i=[(1, 1)])
    assert Rs == [(1, 0, 0)]
    r = o3.rand_angles()
    D = o3.irr_repr(1, *r)
    Q = Q.reshape(3, 3, 3)
    Q1 = torch.einsum('li,mj,nk,ijk', D, D, D, Q)
    assert (Q1 - Q).abs().max() < 1e-10


def test_reduce_tensor_antisymmetric_L2():
    Rs, Q = rs.reduce_tensor('ijk=-ikj=-jik', i=[(1, 2)])
    assert Rs[0] == (1, 1, 0)
    q = Q[:3].reshape(3, 5, 5, 5)

    r = o3.rand_angles()
    D1 = o3.irr_repr(1, *r)
    D2 = o3.irr_repr(2, *r)
    Q1 = torch.einsum('il,jm,kn,zijk->zlmn', D2, D2, D2, q)
    Q2 = torch.einsum('yz,zijk->yijk', D1, q)

    assert (Q1 - Q2).abs().max() < 1e-10
    assert (q + q.transpose(1, 2)).abs().max() < 1e-10
    assert (q + q.transpose(1, 3)).abs().max() < 1e-10
    assert (q + q.transpose(3, 2)).abs().max() < 1e-10


def test_reduce_tensor_elasticity_tensor():
    Rs, _Q = rs.reduce_tensor('ijkl=jikl=klij', i=[(1, 1)])
    assert rs.dim(Rs) == 21


def test_reduce_tensor_elasticity_tensor_parity():
    Rs, _Q = rs.reduce_tensor('ijkl=jikl=klij', i=[(1, 1, -1)])
    assert all(p == 1 for (_, _, p) in Rs)
    assert rs.dim(Rs) == 21
