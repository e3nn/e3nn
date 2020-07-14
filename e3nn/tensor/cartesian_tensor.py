import torch
from e3nn.tensor.irrep_tensor import IrrepTensor
from e3nn import rs, o3


class CartesianTensor():
    def __init__(self, tensor, formula=None):
        self.N = len(tensor.shape)
        if self.N > 13:
            raise ValueError("CartesianTensor only supports up to 13 indices.")
        assert tuple(tensor.shape) == tuple(3 for i in range(self.N))
        if formula is None:
            formula = "abcdefghijklmnopqrstuvxyz"[:self.N]
        self.formula = formula
        self.tensor = tensor

    def to_irrep_tensor(self):
        basis_change = o3.xyz_vector_basis_to_spherical_basis()
        for i in range(self.tensor.dim()):
            irrep_basis = torch.tensordot(basis_change, self.tensor, dims=([1], [i])).transpose(0, i)
        Rs = [(1, 1)]  # vectors
        Rs_out, Q = rs.reduce_tensor(self.formula, **{i: Rs for i in old_indices})
        irrep_tensor = torch.einsum('ab,b->a', Q, irrep_basis.view(-1))
        return IrrepTensor(irrep_tensor, Rs_out)
