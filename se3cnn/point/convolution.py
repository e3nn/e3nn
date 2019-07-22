# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name
# TODO find a better way to do, now it computes diff_matrix and so on at every convolution
import torch

from se3cnn.point.utils import apply_kernel, convolve, difference_matrix


class SE3PointConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, neighbors=None, rel_mask=None):
        """
        :param features: tensor ([batch,] channel, point)
        :param geometry: tensor ([batch,] point, xyz)
        :param neighbors: index tensor ([batch,] point, neighbor)
        :param rel_mask: tensor ([batch,] point_out, point_in)
        :return: tensor ([batch,] channel, point)
        """
        return convolve(self.kernel, features, geometry, neighbors, rel_mask)


class SE3PointApplyKernel(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, neighbors=None, rel_mask=None):
        """
        :param features: tensor ([batch,] channel, point)
        :param geometry: tensor ([batch,] point, xyz)
        :param neighbors: index tensor ([batch,] point, neighbor)
        :param rel_mask: tensor ([batch,] point_out, point_in)
        :return: tensor ([batch,] channel, point, point)
        """
        return apply_kernel(self.kernel, features, geometry, neighbors, rel_mask)


# TODO optimize this class
class SE3PointNeighborsConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out, radius):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)
        self.radius = radius

    def forward(self, features, geometry):
        """
        :param features: tensor [batch, channel, point]
        :param geometry: tensor [batch, point, xyz]
        :return: tensor [batch, channel, point]
        """
        batch, n, _ = geometry.size()
        diff = difference_matrix(geometry).norm(2, dim=-1).view(batch * n, n)  # [batch * a, b]
        neighbors = [
            (diff[i] < self.radius).nonzero().flatten()
            for i in range(batch * n)
        ]
        nn = max(len(nei) for nei in neighbors)
        rel_mask = features.new_zeros(batch * n, nn)
        for i, nei in enumerate(neighbors):
            rel_mask[i, :len(nei)] = 1
        rel_mask = rel_mask.view(batch, n, nn)  # [batch, a, b]
        neighbors = torch.stack([
            torch.cat([nei, nei.new_zeros(nn - len(nei))])
            for nei in neighbors])
        neighbors = neighbors.view(batch, n, nn)  # [batch, a, b]
        return convolve(self.kernel, features, geometry, neighbors, rel_mask)
