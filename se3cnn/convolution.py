# pylint: disable=C, R, arguments-differ, redefined-builtin
import torch

from se3cnn import SE3Kernel
from se3cnn.kernel import gaussian_window_wrapper
from se3cnn.point_utils import convolve


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_in, Rs_out, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):
        return torch.nn.functional.conv3d(input, self.kernel(), **self.kwargs)


class SE3ConvolutionTranspose(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_out, Rs_in, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):
        return torch.nn.functional.conv_transpose3d(input, self.kernel(), **self.kwargs)


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
