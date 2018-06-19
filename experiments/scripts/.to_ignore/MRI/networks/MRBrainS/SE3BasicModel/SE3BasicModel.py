import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from se3cnn.blocks import GatedBlock
from se3cnn import basis_kernels


class network(nn.Module):

    def __init__(self, output_size, filter_size=5):
        super(network, self).__init__()

        features = [(3,),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    (output_size,)]

        common_block_params = {
            'size': filter_size,
            'padding': filter_size//2,
            'stride': 1,
            'normalization': 'instance',
            'radial_window': partial(
                basis_kernels.gaussian_window_fct_convenience_wrapper,
                mode='compromise', border_dist=0, sigma=0.6),
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1],
                             **common_block_params, **block_params[i])
                  for i in range(len(block_params))]

        self.layers = torch.nn.Sequential(
            *blocks,
        )

    def forward(self, x):
        out = self.layers(x)
        return out

