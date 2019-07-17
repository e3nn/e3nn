import torch
import torch.nn as nn


class network(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(network, self).__init__()
        size = filter_size
        bias = True
        self.layers = nn.Sequential(
                        nn.Conv3d(3, 8, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(8),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(8, 16, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(16),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(16, 32, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(32, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128, output_size, kernel_size=1, padding=0, stride=1, bias=bias))

    def forward(self, x):
        out = self.layers(x)
        return out