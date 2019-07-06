# pylint: disable=C,E1101,E1102
import torch


def difference_matrix(geometry):
    ri = geometry.unsqueeze(-2)  # [N, 1, 3]
    rj = geometry.unsqueeze(-3)  # [1, N, 3]
    rij = ri - rj  # [N, N, 3]
    return rij


def relative_mask(mask):
    return torch.einsum('ti,tj->tij', (mask, mask))


def neighbor_difference_matrix(neighbors, geometry):
    """
    :param neighbors: index tensor ([B], N, K)
    :param geometry: tensor ([B], N, 3)
    """
    if neighbors.dim() == 2:
        N, _K = neighbors.size()
        ri = geometry[neighbors, :]  # [N, K, 3]
        rj = geometry.unsqueeze(-2)  # [N, 1, 3]
    elif neighbors.dim() == 3:
        B, N, _K = neighbors.size()
        ri = geometry[torch.arange(B).view(-1, 1, 1), neighbors, :]  # [B, N, K, 3]
        rj = geometry[torch.arange(B).view(-1, 1), torch.arange(N).view(1, -1), :].unsqueeze(-2)  # [B, N, 1, 3]
    rij = ri - rj  # [N, K, 3] or [B, N, K, 3]
    return rij


def neighbor_feature_matrix(neighbors, features):
    """
    Args:
       neighbors: LongTensor of [batch, points, neighbors]
       features: FloatTensor of [batch, channel, points]

    Returns:
       neighbor_features: FloatTensor of [batch, channel, points, neighbors]
    """
    if features.dim() == 3:  # Has batch dimension
        B, _N, _K = neighbors.size()

        features = torch.transpose(features, 0, 1)  # [C, B, N]
        neighbor_features = features[:, torch.arange(B).view(-1, 1, 1), neighbors]  # [C, B, N, K]
        neighbor_features = torch.transpose(neighbor_features, 0, 1)  # [B, C, N, K]
    else:
        neighbor_features = features[:, neighbors]  # [C, N, K]
    return neighbor_features
