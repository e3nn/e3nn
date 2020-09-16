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

    def to_irrep_transformation(self):
        dim = self.tensor.dim()
        change = o3.kron(*[o3.xyz_to_irreducible_basis()] * dim)
        Rs = [(1, 1, -1)]  # vectors
        old_indices = self.formula.split("=")[0]
        Rs_out, Q = rs.reduce_tensor(self.formula, **{i: Rs for i in old_indices})
        return Rs_out, torch.einsum('ab,bc->ac', Q, change.reshape(3 ** dim, 3 ** dim))

    def to_irrep_tensor(self):
        Rs_out, Q = self.to_irrep_transformation()
        tensor = torch.einsum('ab,b->a', Q, self.tensor.reshape(-1))
        return IrrepTensor(tensor, Rs_out)
