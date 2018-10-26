# pylint: disable=C,R,E1101,E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np
from scipy.ndimage import zoom

from experiments.util import lr_schedulers

from se3cnn.blocks import GatedBlock
from se3cnn.SE3 import rotate_scalar
from se3cnn.SO3 import rot
from se3cnn.filter import low_pass_filter


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class LowPass(nn.Module):
    def __init__(self, scale, stride):
        super().__init__()
        self.scale = scale
        self.stride = stride

    def forward(self, inp):
        return low_pass_filter(inp, self.scale, self.stride)


def get_volumes(size=20, pad=8, rotate=False, rotate90=False):
    assert size >= 4
    tetris_tensorfields = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]

    labels = np.arange(len(tetris_tensorfields))

    tetris_vox = []
    for shape in tetris_tensorfields:
        volume = np.zeros((4, 4, 4))
        for xi_coords in shape:
            volume[xi_coords] = 1

        volume = zoom(volume, size / 4, order=0)
        volume = np.pad(volume, pad, 'constant')

        if rotate:
            a, c = np.random.rand(2) * 2 * np.pi
            b = np.arccos(np.random.rand() * 2 - 1)
            volume = rotate_scalar(volume, rot(a, b, c))
        if rotate90:
            volume = rot_volume_90(volume)

        tetris_vox.append(volume[np.newaxis, ...])

    tetris_vox = np.stack(tetris_vox).astype(np.float32)

    return tetris_vox, labels


def rot_volume_90(vol):
    k1, k2, k3 = np.random.randint(4, size=3)
    vol = np.rot90(vol, k=k1, axes=(0, 1))  # z
    vol = np.rot90(vol, k=k2, axes=(0, 2))  # y
    vol = np.rot90(vol, k=k3, axes=(0, 1))  # z
    return vol


def train(network, dataset, N_epochs):
    network.train()

    volumes, labels = dataset
    volumes = torch.tensor(volumes).cuda()
    labels = torch.tensor(labels).cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    for epoch in range(N_epochs):
        predictions = network(volumes)

        optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, init_lr=1e-1, epoch_start=100, base_factor=.98, verbose=False)

        optimizer.zero_grad()
        F.cross_entropy(predictions, labels).backward()
        optimizer.step()

        argmax = predictions.argmax(1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('epoch {}: acc={:.2f}'.format(epoch, acc), end=" \r")


def test(network, dataset):
    network.eval()

    volumes, labels = dataset
    volumes = torch.tensor(volumes).cuda()
    labels = torch.tensor(labels).cuda()

    predictions = network(volumes)
    argmax = predictions.argmax(1)
    acc = (argmax.squeeze() == labels).float().mean().item()

    print('test acc={:.2f}'.format(acc), end=" \r")
    return acc


class SE3Net(torch.nn.Module):

    def __init__(self, smooth_stride):
        super().__init__()

        features = [
            (1,),
            (4, 4, 4, 1),
            (16, 16, 16, 0),
            (32, 16, 16, 0),
            (128,)
        ]
        common_block_params = {
            'size': 5,
            'padding': 4,
            'activation': (F.relu, torch.sigmoid),
            'smooth_stride': smooth_stride,
        }
        block_params = [
            { 'stride': 1 },
            { 'stride': 2 },
            { 'stride': 2 },
            { 'stride': 1 },
        ]

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(features) - 1)
        ]
        self.sequence = torch.nn.Sequential(*blocks,
                                            AvgSpacial(),
                                            nn.Dropout(p=.2),
                                            nn.Linear(features[-1][0], 10))

    def forward(self, inp):  # pylint: disable=W
        inp = low_pass_filter(inp, 2)
        return self.sequence(inp)



class CNN(torch.nn.Module):

    def __init__(self, smooth_stride):
        super().__init__()

        features = [
            1,
            43,
            144,
            160,
            128,
        ]
        common_block_params = {
            'kernel_size': 5,
            'padding': 4,
        }
        block_params = [
            { 'stride': 1 },
            { 'stride': 2 },
            { 'stride': 2 },
            { 'stride': 1 },
        ]

        if smooth_stride:
            blocks = [
                torch.nn.Sequential(
                    nn.Conv3d(features[i], features[i + 1], **common_block_params),
                    LowPass(block_params[i]['stride'], block_params[i]['stride']),
                    nn.BatchNorm3d(features[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(features) - 1)
            ]
        else:
            blocks = [
                torch.nn.Sequential(
                    nn.Conv3d(features[i], features[i + 1], **common_block_params, **block_params[i]),
                    nn.BatchNorm3d(features[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(features) - 1)
            ]

        self.sequence = torch.nn.Sequential(*blocks,
                                            AvgSpacial(),
                                            nn.Dropout(p=.2),
                                            nn.Linear(features[-1], 10))

    def forward(self, inp):  # pylint: disable=W
        inp = low_pass_filter(inp, 2)
        return self.sequence(inp)


def experiment(network):
    N_epochs = 250
    N_test = 100
    trainset = get_volumes(rotate=True)  # train with randomly rotated pieces but only once

    train(network, trainset, N_epochs=N_epochs)
    test_accs = []
    for _ in range(N_test):
        testset = get_volumes(rotate=True)
        acc = test(network, testset)
        test_accs.append(acc)

    return np.mean(test_accs)

def main():
    torch.backends.cudnn.benchmark = True

    for smooth_stride in [True, False]:
        for Model in [SE3Net, CNN]:
            for _rep in range(12):
                network = Model(smooth_stride).cuda()
                acc = experiment(network)
                print("smooth_stride={} model={} acc= {}".format(smooth_stride, Model, acc))


main()
