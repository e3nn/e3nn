# pylint: disable=C,E1101,E1102
import torch
from functools import partial


def _apply(equation, equation_neighbors, kernel, features, geometry, neighbors=None, rel_mask=None):
    """
    :param kernel: kernel model
    :param features: tensor [[batch,] channel, point]
    :param geometry: tensor [[batch,] point, xyz]
    :param neighbors: index tensor [[batch,] point, neighbor]
    :param rel_mask: tensor [[batch,] point_out, point_in]
    :return: tensor [[batch,] ...]
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

    kernel = kernel(diff_matrix)  # [batch, point_out, point_in, channel_out, channel_in]

    if rel_mask is not None:
        kernel = torch.einsum('zab,zabij->zabij', (rel_mask, kernel))

    if neighbors is None:
        output = torch.einsum(equation, (features, kernel))  # [batch, ...]
    else:
        output = torch.einsum(equation_neighbors, (features, kernel))  # [batch, ...]

    if not has_batch:
        output = output.squeeze(0)

    return output


convolve = partial(_apply, 'zjb,zabij->zia', 'zjab,zabij->zia')
apply_kernel = partial(_apply, 'zjb,zabij->ziab', 'zjab,zabij->ziab')


def difference_matrix(geometry):
    rb = geometry.unsqueeze(-3)  # [1, N, 3]
    ra = geometry.unsqueeze(-2)  # [N, 1, 3]
    rab = rb - ra  # [N, N, 3]
    return rab


def relative_mask(mask):
    return torch.einsum('ti,tj->tij', (mask, mask))


def neighbor_difference_matrix(neighbors, geometry):
    """
    :param neighbors: index tensor ([B], N, K)
    :param geometry: tensor ([B], N, 3)
    """
    if neighbors.dim() == 2:
        N, _K = neighbors.size()
        rb = geometry[neighbors, :]  # [N, K, 3]
        ra = geometry.unsqueeze(-2)  # [N, 1, 3]
    elif neighbors.dim() == 3:
        B, N, _K = neighbors.size()
        rb = geometry[torch.arange(B).view(-1, 1, 1), neighbors, :]  # [B, N, K, 3]
        ra = geometry[torch.arange(B).view(-1, 1), torch.arange(N).view(1, -1), :].unsqueeze(-2)  # [B, N, 1, 3]
    rab = rb - ra  # [N, K, 3] or [B, N, K, 3]
    return rab


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
