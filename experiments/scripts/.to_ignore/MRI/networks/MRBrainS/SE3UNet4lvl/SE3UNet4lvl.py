import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from se3cnn import basis_kernels
from experiments.util.arch_blocks import NonlinearityBlock
from experiments.util.arch_blocks import SkipSumBlock



class network(nn.Module):
    def __init__(self, output_size, args):
        super(network, self).__init__()

        common_params = {
            'radial_window': partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'size': args.kernel_size,
            'padding': args.kernel_size//2,
            'activation':(F.relu, F.sigmoid),
            'normalization': args.normalization,
            'capsule_dropout_p': args.p_drop_conv,
            'SE3_nonlinearity': args.SE3_nonlinearity,
            'batch_norm_momentum': 0.01,
        }

        features = [(3,), # in
                    ( 16, 12,  8,  8),     # level 1 (enc and dec)
                    ( 32, 24, 16,  8, 4), # level 2 (enc and dec)
                    ( 64, 48, 32, 16, 8), # level 3 (enc and dec)
                    ( 64, 48, 32, 16, 8), # level 4 (bridge)
                    (512,), # 1x1 conv
                    # ( 8,  8,  4),     # level 1 (enc and dec)
                    # (16,  8,  4), # level 2 (enc and dec)
                    # (32, 16,  8), # level 3 (enc and dec)
                    # (48, 32, 16), # level 4 (bridge)
                    # (256,), # 1x1 conv
                    (output_size,)] # out

        # encoder pathway
        self.enc1 = nn.Sequential(
            NonlinearityBlock(features[0], features[1], stride=1, **common_params),
            NonlinearityBlock(features[1], features[1], stride=1, **common_params))
        self.enc2 = nn.Sequential(
            NonlinearityBlock(features[1], features[2], stride=2, **common_params),
            NonlinearityBlock(features[2], features[2], stride=1, **common_params))
        self.enc3 = nn.Sequential(
            NonlinearityBlock(features[2], features[3], stride=2, **common_params),
            NonlinearityBlock(features[3], features[3], stride=1, **common_params))

        # bridge
        self.bridge = nn.Sequential(
            NonlinearityBlock(features[3], features[4], stride=2, **common_params),
            NonlinearityBlock(features[4], features[4], stride=1, **common_params),
            NonlinearityBlock(features[4], features[4], stride=1, **common_params),
            nn.Upsample(scale_factor=2, mode="nearest"),
            NonlinearityBlock(features[4], features[3], stride=1, **common_params))

        # skip connection with convolution and summation
        self.merge3 = SkipSumBlock(features[3], **common_params)
        self.merge2 = SkipSumBlock(features[2], **common_params)
        self.merge1 = SkipSumBlock(features[1], **common_params)

        # decoder pathway
        self.dec3 = nn.Sequential(
            NonlinearityBlock(features[3], features[3], stride=1, **common_params),
            nn.Upsample(scale_factor=2, mode="nearest"),
            NonlinearityBlock(features[3], features[2], stride=1, **common_params))
        self.dec2 = nn.Sequential(
            NonlinearityBlock(features[2], features[2], stride=1, **common_params),
            nn.Upsample(scale_factor=2, mode="nearest"),
            NonlinearityBlock(features[2], features[1], stride=1, **common_params))
        self.dec1 = nn.Sequential(
            NonlinearityBlock(features[1], features[ 1], stride=1, **common_params),
            NonlinearityBlock(features[1], features[-2], stride=1, **common_params))

        # 1x1 conv
        self.drop_final = nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None
        self.conv_final = nn.Conv3d(int(features[-2][0]), int(features[-1][0]), kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        # encoder path
        enc1_out   = self.enc1(x)
        enc2_out   = self.enc2(enc1_out)
        enc3_out   = self.enc3(enc2_out)
        # bridge
        bridge_out = self.bridge(enc3_out)
        # skip connections and decoder
        merge3_out = self.merge3(enc=enc3_out, dec=bridge_out)
        dec3_out   = self.dec3(merge3_out)
        merge2_out = self.merge2(enc=enc2_out, dec=dec3_out)
        dec2_out   = self.dec2(merge2_out)
        merge1_out = self.merge1(enc=enc1_out, dec=dec2_out)
        dec1_out   = self.dec1(merge1_out)
        # 1x1 convolution mapping scalar capsules to classes
        if self.drop_final is not None:
            dec1_out = self.drop_final(dec1_out)
        out = self.conv_final(dec1_out)
        return out