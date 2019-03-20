import torch

def difference_matrix(geometry):
    ri = geometry.unsqueeze(-2)  # [N, 1, 3]
    rj = geometry.unsqueeze(-3)  # [1, N, 3]
    rij = ri - rj  # [N, N, 3]
    return rij

def relative_mask(mask):
    return torch.einsum('ti,tj->tij', (mask, mask))
