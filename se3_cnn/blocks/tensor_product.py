# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3BNConvolution
from se3_cnn.non_linearities import ScalarActivation
from se3_cnn.non_linearities import TensorProduct
from se3_cnn import SO3


class TensorProductBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, relu, size, n_radial, stride=1, padding=0, batch_norm_momentum=0.1):
        super().__init__()
        self.tensor = TensorProduct([(repr_in[0], 1, False), (repr_in[1], 3, True), (repr_in[2], 5, False)]) if repr_in[1] > 0 else None
        self.bn_conv = SE3BNConvolution(
            size=size,
            n_radial=n_radial,
            Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5), (repr_in[1], SO3.repr3x3)],
            Rs_out=[(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)],
            stride=stride,
            padding=padding,
            momentum=batch_norm_momentum,
            mode='maximum')

        if relu:
            self.relu = ScalarActivation([(repr_out[0], True), (repr_out[1] * 3, False), (repr_out[2] * 5, False)], torch.nn.functional.relu)
        else:
            self.relu = None

    def forward(self, sv5):  # pylint: disable=W
        if self.tensor is not None:
            t = self.tensor(sv5)
            sv5t = torch.cat([sv5, t], dim=1)
        else:
            sv5t = sv5

        sv5 = self.bn_conv(sv5t)

        if self.relu is not None:
            sv5 = self.relu(sv5)
        return sv5
