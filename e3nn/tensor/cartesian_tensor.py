import torch
from e3nn.tensor.irrep_tensor import IrrepTensor
from e3nn import rs, o3


class CartesianTensor():
    def __init__(self, tensor, formula=None):
        if tuple(tensor.shape) != tuple(3 for i in range(tensor.dim())):
            raise ValueError(f"all dimensions of tensor should have shape 3 but tensor has shape {tensor.shape}")
        if formula is None:
            formula = "abcdefghijklmnopqrstuvxyz"[:tensor.dim()]
        self.formula = formula
        self.tensor = tensor

    def to_irrep_tensor(self):
        basis_change = o3.xyz_to_irreducible_basis()
        tensor = self.tensor
        for i in range(self.tensor.dim()):
            tensor = torch.tensordot(basis_change, tensor, dims=([1], [i])).transpose(0, i)
        Rs = [(1, 1)]  # vectors
        old_indices = self.formula.split("=")[0]
        Rs_out, Q = rs.reduce_tensor(self.formula, **{i: Rs for i in old_indices})
        tensor = torch.einsum('ab,b->a', Q, tensor.reshape(-1))
        return IrrepTensor(tensor, Rs_out)
