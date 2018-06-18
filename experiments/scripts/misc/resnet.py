import torch.nn as nn
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock


class ResNetBlock(nn.Module):
    def __init__(self, features, stride=1):
        super().__init__()

        n = len(features) - 1

        self.sequential = nn.Sequential(*[
            GatedBlock(features[i], features[i + 1],
                       size=7,
                       padding=3,
                       stride=stride if i == 0 else 1,
                       n_radial=2,
                       activation=F.relu)
            for i in range(n)])

        self.shortcut = GatedBlock(features[0], features[-1],
                                   size=1,
                                   stride=stride,
                                   n_radial=1,
                                   activation=None)

    def forward(self, x):
        a = self.sequential(x)
        b = self.shortcut(x)
        return a + b


class AvgSpacial(nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


def ResNet():
    return nn.Sequential(
        GatedBlock((1,), (5, 3, 1), size=7, n_radial=2, activation=F.relu),
        ResNetBlock([(5, 3, 1), (1, 1, 1), (5, 3, 1)], stride=2),
        ResNetBlock([(5, 3, 1), (1, 1, 1), (5, 3, 1)], stride=2),
        ResNetBlock([(5, 3, 1), (1, 1, 1), (2,)], stride=2),
        AvgSpacial()
    )
