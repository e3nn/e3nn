# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch

from e3nn.point.kernel import Kernel
from e3nn.point.radial import ConstantRadialModel


class Linear(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        super().__init__()

        # hack: use Kernel to construct the weight matrix
        def get_l_filters(l_in, l_out):
            return [0] if l_in == l_out else []
        self.kernel = Kernel(Rs_in, Rs_out, ConstantRadialModel, get_l_filters)

    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensor [..., channel]
        """
        *size, n = features.size()
        features = features.view(-1, n)

        k = self.kernel(features.new_zeros(3))
        features = torch.einsum("ij,zj->zi", k, features)
        features = features.view(*size, -1)
        return features
