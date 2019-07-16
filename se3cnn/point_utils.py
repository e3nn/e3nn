# pylint: disable=C,E1101,E1102
import torch


def convolve(kernel, features, geometry, neighbors=None, rel_mask=None):
    """
    :param kernel: kernel model
    :param features: tensor ([batch,] channel, point)
    :param geometry: tensor ([batch,] point, xyz)
    :param neighbors: index tensor ([batch,] point, neighbor)
    :param rel_mask: tensor ([batch,] point_out, point_in)
    :return: tensor ([batch,] channel, point)
    """

    has_batch = features.dim() == 3

    if not has_batch:
        features = features.unsqueeze(0)
        geometry = geometry.unsqueeze(0)
        neighbors = neighbors.unsqueeze(0) if neighbors is not None else None
        rel_mask = rel_mask.unsqueeze(0) if rel_mask is not None else None

    if neighbors is None:
        diff_matrix = difference_matrix(geometry)  # [batch, point_out, point_in, 3]
    else:
        diff_matrix = neighbor_difference_matrix(neighbors, geometry)  # [batch, point_out, point_in, 3]
        features = neighbor_feature_matrix(neighbors, features)  # [batch, channel, point_out, point_in]

    kernel = kernel(diff_matrix)  # [channel_out, channel_in, batch, point_out, point_in]

    if rel_mask is not None:
        kernel = torch.einsum('nba,dcnba->dcnba', (rel_mask, kernel))

    if neighbors is None:
        output = torch.einsum('nca,dcnba->ndb', (features, kernel))  # [batch, channel, point]
    else:
        output = torch.einsum('ncba,dcnba->ndb', (features, kernel))  # [batch, channel, point]

    if not has_batch:
        output = output.squeeze(0)

    return output


def difference_matrix(geometry):
    ri = geometry.unsqueeze(-3)  # [1, N, 3]
    rj = geometry.unsqueeze(-2)  # [N, 1, 3]
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
