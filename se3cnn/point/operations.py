# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long
# TODO find a better way to do, now it computes diff_matrix and so on at every convolution
import torch


class Convolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, channel]
        """
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra)  # [batch, a, b, i, j]
        return torch.einsum("zabij,zbj->zai", (k, features))  # [batch, point, channel]


class ApplyKernel(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, point, channel]
        """
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra)  # [batch, a, b, i, j]
        return torch.einsum("zabij,zbj->zabi", (k, features))  # [batch, point, point, channel]


# TODO optimize this class
class NeighborsConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out, radius):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)
        self.radius = radius

    def forward(self, features, geometry):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, channel]
        """
        batch, n, _ = geometry.size()

        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        diff = rb - ra  # [batch, a, b, xyz]
        norm = diff.norm(2, dim=-1).view(batch * n, n)  # [batch * a, b]

        neighbors = [
            (norm[i] < self.radius).nonzero().flatten()
            for i in range(batch * n)
        ]

        k = max(len(nei) for nei in neighbors)
        rel_mask = features.new_zeros(batch * n, k)
        for i, nei in enumerate(neighbors):
            rel_mask[i, :len(nei)] = 1
        rel_mask = rel_mask.view(batch, n, k)  # [batch, a, b]

        neighbors = torch.stack([
            torch.cat([nei, nei.new_zeros(k - len(nei))])
            for nei in neighbors
        ])
        neighbors = neighbors.view(batch, n, k)  # [batch, a, b]

        rb = geometry[torch.arange(batch).view(-1, 1, 1), neighbors, :]  # [batch, a, b, xyz]
        ra = geometry[torch.arange(batch).view(-1, 1, 1), torch.arange(n).view(1, -1, 1), :]  # [batch, a, 1, xyz]
        diff = rb - ra  # [batch, a, b, xyz]

        neighbor_features = features[torch.arange(batch).view(-1, 1, 1), neighbors, :]  # [batch, a, b, j]

        kernel = self.kernel(diff)  # [batch, a, b, i, j]
        output = torch.einsum('zab,zabij,zabj->zai', (rel_mask, kernel, neighbor_features))  # [batch, a, i]

        return output
