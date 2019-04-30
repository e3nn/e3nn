import torch

def difference_matrix(geometry):
    ri = geometry.unsqueeze(-2)  # [N, 1, 3]
    rj = geometry.unsqueeze(-3)  # [1, N, 3]
    rij = ri - rj  # [N, N, 3]
    return rij

def get_relative_mask(mask):
    return torch.einsum('ti,tj->tij', (mask, mask))

def neighbor_difference_matrix(neighbors, geometry):
    N, K = neighbors.shape[-2:]
    ri = geometry[..., neighbors, :]  # [N, K, 3]
    rj = geometry[..., torch.arange(N), :].unsqueeze(-2)  # [N, 1, 3]
    rij = ri - rj  # [N, K, 3]
    return rij
    
def neighbor_feature_matrix(neighbors, features):
    """
    Args:
       neighbors: LongTensor of [batch, points, neighbors]
       features: FloatTensor of [batch, channel, points]
    
    Returns:
       neighbor_features: FloatTensor of [batch, channel, points, neighbors]
    """
    if len(features.shape) == 3:  # Has batch dimension
        features = torch.transpose(features, 0, 1)
    neighbor_features = features[..., neighbors]  # [C, N, K] or [C, B, N, K]
    if len(features.shape) == 3:  # Has batch dimension
        neighbor_features = torch.transpose(neighbor_features, 0, 1)
    return neighbor_features
