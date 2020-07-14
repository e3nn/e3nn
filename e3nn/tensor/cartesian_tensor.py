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
        basis_change = o3.xyz_vector_basis_to_spherical_basis()
        new_indices = self.base_indices[self.N:2 * self.N]
        old_indices = self.base_indices[:self.N]
        rotation_pairs = "".join([a + b + "," for a, b in zip(new_indices, old_indices)])
        einsum_str = rotation_pairs + old_indices + "->" + new_indices
        irrep_basis = torch.einsum(einsum_str, *[basis_change] * self.N, self.tensor)
        Rs = [(1, 1)]  # vectors
        Rs_out, Q = rs.reduce_tensor(self.formula, **{i: Rs for i in old_indices})
        irrep_tensor = torch.einsum('ab,b->a', Q, irrep_basis.view(-1))
        return IrrepTensor(irrep_tensor, Rs_out)
