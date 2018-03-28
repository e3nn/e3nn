# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3Convolution
from se3_cnn import SE3BatchNorm
from se3_cnn import SO3


class SE3BNConvolution(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3BatchNorm followed by SE3Convolution
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()
        self.bn = SE3BatchNorm([(m, SO3.dim(R)) for m, R in Rs_in], eps=eps, momentum=momentum)
        self.conv = SE3Convolution(Rs_in=Rs_in, Rs_out=Rs_out, size=size, radial_window_dict=radial_window_dict, **kwargs)

    def forward(self, input):  # pylint: disable=W
        return self.conv(self.bn(input))


class SE3ConvolutionBN(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3Convolution followed by SE3BatchNorm
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()
        self.conv = SE3Convolution(Rs_in=Rs_in, Rs_out=Rs_out, size=size, radial_window_dict=radial_window_dict, **kwargs)
        self.bn = SE3BatchNorm([(m, SO3.dim(R)) for m, R in Rs_out], eps=eps, momentum=momentum)

    def forward(self, input):  # pylint: disable=W
        return self.bn(self.conv(input))
