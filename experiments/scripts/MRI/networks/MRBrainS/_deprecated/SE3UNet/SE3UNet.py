import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from se3cnn.blocks import GatedBlock
from se3cnn import basis_kernels
from experiments.util.arch_blocks import Merge


class network(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(network, self).__init__()
        size = filter_size

        common_params = {
            'radial_window': partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                     mode='compromise', border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
        }

        features = [(3,),
                    ( 8,  8,  8,  4),
                    (16, 16, 16,  8),
                    (32, 32, 32, 16),
                    (16, 16, 16,  8),
                    ( 8,  8,  8,  4),
                    (output_size,)]

        # TODO: do padding using ReplicationPad3d?
        # TODO: on validation - use overlapping patches and only use center of patch

        self.conv1 = nn.Sequential(
            GatedBlock(features[0], features[1], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[1], features[1], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv2 = nn.Sequential(
            GatedBlock(features[1], features[2], size=size, padding=size//2, stride=2, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[2], features[2], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv3 = nn.Sequential(
            GatedBlock(features[2], features[3], size=size, padding=size//2, stride=2, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[3], features[3], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedBlock(features[3], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.merge1 = Merge()

        self.conv4 = nn.Sequential(
            GatedBlock(features[3], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[4], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedBlock(features[4], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.merge2 = Merge()

        self.conv5 = nn.Sequential(
            GatedBlock(features[4], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[5], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv_final = GatedBlock(features[5], features[6], size=1, padding=0, stride=1, activation=None, normalization=None, **common_params)

    def forward(self, x):

        conv1_out  = self.conv1(x)
        conv2_out  = self.conv2(conv1_out)
        conv3_out  = self.conv3(conv2_out)
        up1_out    = self.up1(conv3_out)
        merge1_out = self.merge1(conv2_out, up1_out)
        conv4_out  = self.conv4(merge1_out)
        up2_out    = self.up2(conv4_out)
        merge2_out = self.merge2(conv1_out, up2_out)
        conv5_out  = self.conv5(merge2_out)
        out        = self.conv_final(conv5_out)

        return out