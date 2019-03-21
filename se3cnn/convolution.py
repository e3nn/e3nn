# pylint: disable=C,R,E1101,E1102
import torch
from se3cnn import SE3Kernel
from se3cnn.point_kernel import SE3PointKernel
from se3cnn import kernel, point_kernel


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=kernel.gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_in, Rs_out, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):  # pylint: disable=W
        return torch.nn.functional.conv3d(input, self.kernel(), **self.kwargs)


class SE3ConvolutionTranspose(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=kernel.gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_out, Rs_in, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):  # pylint: disable=W
        return torch.nn.functional.conv_transpose3d(input, self.kernel(), **self.kwargs)


class SE3PointConvolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, radii, radial_function=point_kernel.gaussian_radial_function, J_filter_max=10, kernel=SE3PointKernel, **kwargs):
        super().__init__()

        self.kernel = kernel(Rs_in, Rs_out, radii, radial_function=radial_function, J_filter_max=J_filter_max)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input, difference_mat, relative_mask=None):  # pylint: disable=W
        kernel = self.kernel(difference_mat)

        if len(input.size()) == 2:
            # No batch dimension
            output = torch.einsum('ca,dcba->db', (input, kernel))
        elif len(input.size()) == 3:
            # Batch dimension
            # Apply relative_mask to kernel (if examples are not all size N, M)
            if relative_mask is not None:
                kernel = torch.einsum('nba,dcnba->dcnba', (relative_mask, kernel))
            output = torch.einsum('nca,dcnba->ndb', (input, kernel))

        return output

