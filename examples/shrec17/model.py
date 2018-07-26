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

    def __init__(self, n_out):
        super().__init__()

        self.int_repr = None

        features = [  # (6) double all channels
            (1, ),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 16, 16),
            (200, )
        ]

        common_block_params = {
            'size': 5,  # (5) = 5->7
            'stride': 2,
            'padding': 3,
            'normalization': 'batch',
            'capsule_dropout_p': 0.1  # (4) Maurice suggestion
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
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
