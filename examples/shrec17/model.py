# pylint: disable=no-member, missing-docstring
import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.image.gated_block import GatedBlock


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class Model(torch.nn.Module):

    def __init__(self, n_out):
        super().__init__()

        self.int_repr = None

        features = [
            (1, ), # 64
            (8, 4, 2), (8, 4, 2), # 34, 38
            (16, 8, 4), (16, 8, 4), # 21, 25
            (32, 16, 8), (32, 16, 8), # 15, 19
            (32, 16, 8), # 12
            (512, )
        ]

        common_block_params = {
            'size': 5,
            'padding': 4,
            'normalization': 'batch',
            'smooth_stride': False,
        }

        block_params = [
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid)},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            nn.Linear(features[-1][0], n_out),
        )

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        return self.sequence(x)
