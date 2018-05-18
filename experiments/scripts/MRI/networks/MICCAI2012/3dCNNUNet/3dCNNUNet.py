import torch
import torch.nn as nn

from experiments.util.arch_blocks import Merge


class network(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(network, self).__init__()
        size = filter_size
        bias = True

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=size, padding=size//2, stride=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128))

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=size, padding=size//2, stride=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(256, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True))

        self.merge1 = Merge()

        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(128, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True))

        self.merge2 = Merge()

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=size, padding=size // 2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))

        self.conv_final = nn.Conv3d(64, output_size, kernel_size=1, padding=0, stride=1, bias=bias)


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