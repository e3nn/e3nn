# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
import pytest

from e3nn.tensor.cartesian_tensor import CartesianTensor
torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize('i, j, k', [[0, 1, 2], [0, 2, 1], [1, 2, 3]])
def test_to_irrep(i, j, k):
    # Check antisymmetric component of 3x3 matrix to irrep tensor
    test_M = torch.zeros(3, 3)
    test_M[i, j] = 1
    test_M[j, i] = -1

    cart = CartesianTensor(test_M)
    irrep_tensor = cart.to_irrep_tensor()
    assert irrep_tensor.tensor.nonzero(as_tuple=False).reshape(-1) == torch.LongTensor([k])


def test_user_formula():
    mat = torch.rand(3, 3)
    symm_mat = (mat + mat.transpose(1, 0)) / 2.
    cart = CartesianTensor(symm_mat, "ij=ji")
    irrep_tensor = cart.to_irrep_tensor()
    assert irrep_tensor.Rs == [(1, 0, 1), (1, 2, 1)]
