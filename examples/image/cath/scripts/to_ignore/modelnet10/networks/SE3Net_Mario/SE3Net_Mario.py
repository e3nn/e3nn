import torch.nn as nn
import torch.nn.functional as F
from se3cnn import basis_kernels
from functools import partial
from se3cnn.blocks import GatedBlock
from experiments.util.arch_blocks import *

class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()

        features = [
            ( 1,) if args.add_z_axis is False else (2,),
            ( 4,  4,  4),
            ( 8,  8,  8),
            (16, 16, 16),
            (32, 32, 32),
            (256,),
            (10,)
        ]

        if args.SE3_nonlinearity == 'gated':
            activation = (F.relu, F.sigmoid)
        else:
            activation = F.relu

        radial_window = partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                mode=args.bandlimit_mode, border_dist=0, sigma=0.6)
        common_block_params = {
            'radial_window': radial_window,
            'size': args.kernel_size,
            'padding': args.kernel_size//2,
            'activation': activation,
            'normalization': args.normalization,
            'capsule_dropout_p': args.p_drop_conv,
            'SE3_nonlinearity': args.SE3_nonlinearity,
            'batch_norm_momentum': 0.01
        }
        block_params = [
            {'stride': 2},
            {'stride': 2},
            {'stride': 2},
            {'stride': 2},
            {'stride': 2},
        ]

        blocks = [NonlinearityBlock(features[i], features[i+1], **common_block_params, **block_params[i])
                                                                        for i in range(len(block_params))]

        self.sequence = nn.Sequential(
            *blocks,
            AvgSpacial()
        )
        self.drop_final = nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None
        self.linear_final = nn.Linear(features[-2][0], features[-1][0])


    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = self.sequence(inp)  # [batch, features]
        if self.drop_final is not None:
            x = self.drop_final(x)
        x = self.linear_final(x)
        return x
