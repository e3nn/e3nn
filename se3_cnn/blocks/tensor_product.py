# pylint: disable=C,R,E1101
import torch
from se3_cnn.convolution import SE3Convolution
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn.non_linearities.tensor_product import TensorProduct
from se3_cnn import SO3


class TensorProductBlock(torch.nn.Module):
    def __init__(self, repr_in, repr_out, relu, stride):
        super().__init__()
        self.tensor = TensorProduct([(repr_in[0], 1, False), (repr_in[1], 3, True), (repr_in[2], 5, False)]) if repr_in[1] > 0 else None
        self.conv = SE3Convolution(7, 3,
            [(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)],
            [(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5), (repr_in[1], SO3.repr3x3)],
            stride=stride,
            padding=3)
        self.bn = SE3BatchNorm([(repr_out[0], 1), (repr_out[1], 3), (repr_out[2], 5)])
        self.relu = BiasRelu([(repr_out[0], True), (repr_out[1] * 3, False), (repr_out[2] * 5, False)]) if relu else None

    def forward(self, sv5): # pylint: disable=W
        if self.tensor is not None:
            t = self.tensor(sv5)
            sv5t = torch.cat([sv5, t], dim=1)
        else:
            sv5t = sv5
        sv5 = self.conv(sv5t) # only convolution
        sv5 = self.bn(sv5)
        if self.relu is not None:
            sv5 = self.relu(sv5)
        return sv5
