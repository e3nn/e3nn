# pylint: disable=arguments-differ, redefined-builtin, missing-docstring
import torch
from se3cnn.point.utils import apply_kernel, convolve


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
