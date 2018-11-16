# pylint: disable=C,R,E1101,E1102
import torch
from se3cnn import SE3Kernel
from se3cnn import kernel


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
