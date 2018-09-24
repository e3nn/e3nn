# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.blocks import GatedBlock


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class Model(torch.nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.int_repr = None

        features = [
            (n_in, ), # 77
            (6, 3, 1), (6, 3, 1), # 41, 45
            (14, 6, 3), (14, 6, 3), # 25, 29
            (30, 12, 6), (30, 12, 6), # 17, 21
            (32, 16, 8), # 13
            (512, )
        ]

        common_block_params = {
            'size': 5,
            'padding': 4,
            'normalization': 'batch',
            'smooth_stride': True,
            'activation': (F.relu, torch.sigmoid),
        }

        block_params = [
            {'stride': 2},
            {},
            {'stride': 2},
            {},
            {'stride': 2},
            {},
            {'stride': 2},
            {},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(block_params))
        ]

        linear = nn.Linear(features[-1][0], n_out)
        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            linear,
        )
        nn.init.normal_(linear.weight, std=1 / features[-1][0] ** 0.5)
        nn.init.zeros_(linear.bias)


    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        return self.sequence(x)
