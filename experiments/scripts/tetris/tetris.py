# pylint: disable=C,R,E1101,E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np

from se3_cnn.blocks import GatedBlock
from experiments.util.arch_blocks import AvgSpacial


def get_volumes(size=4, rotate=False):
    assert size >= 4
    tetris_tensorfields = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
                           [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 0)],  # chiral_shape_2
                           [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
                           [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
                           [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
                           [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
                           [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
                           [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # zigzag
    labels = np.arange(len(tetris_tensorfields))
    tetris_vox = []
    for shape in tetris_tensorfields:
        volume = np.zeros((size, size, size))
        shape_slicing = [xi_coords for xi_coords in np.array(shape).T]
        volume[shape_slicing] = 1
        if rotate:
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
    volumes = torch.tensor(volumes)
    labels = torch.tensor(labels)
    # optimizer = optimizers_L1L2.Adam(network.parameters(), 1e-1)
    optimizer = torch.optim.Adam(network.parameters(), 1e-1, weight_decay=1e-5)
    optimizer.zero_grad()
    for epoch in range(N_epochs):
        predictions = network(volumes)
        losses = F.cross_entropy(predictions, labels, reduce=False)
        loss = losses.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, argmax = torch.max(predictions, 1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('epoch {}: acc={}'.format(epoch, acc))


def test(network, dataset):
    network.eval()
    volumes, labels = dataset
    volumes = torch.tensor(volumes)
    labels = torch.tensor(labels)
    predictions = network(volumes)
    _, argmax = torch.max(predictions, 1)
    acc = (argmax.squeeze() == labels).float().mean().item()
    print('test acc={}'.format(acc))
    return acc


class SE3Net(torch.nn.Module):
    def __init__(self):
        super(SE3Net, self).__init__()
        features = [
            (1, ),
            (2, 2, 2, 2),
            (4, 4, 4, 4),
            (16, )
        ]
        common_block_params = {
            'size': 3,
            'stride': 1,
            'padding': 1,
            'capsule_dropout_p': 0.01,
        }
        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
        ]
        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]
        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(16, 10))

    def forward(self, inp):  # pylint: disable=W
        return self.sequence(inp)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.sequence = torch.nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            AvgSpacial(),
            torch.nn.Linear(16, 10))

    def forward(self, inp):  # pylint: disable=W
        return self.sequence(inp)


def main():
    N_epochs = 100
    N_test = 100
    size = 4
    trainset = get_volumes(size)

    network = SE3Net()
    train(network, trainset, N_epochs=N_epochs)
    se3_test_accs = []
    for _ in range(N_test):
        testset = get_volumes(size, rotate=True)
        acc = test(network, testset)
        se3_test_accs.append(acc)

    network = CNN()
    train(network, trainset, N_epochs=N_epochs)
    cnn_test_accs = []
    for _ in range(N_test):
        testset = get_volumes(size, rotate=True)
        acc = test(network, testset)
        cnn_test_accs.append(acc)

    print('avg test acc SE3: {}'.format(np.mean(se3_test_accs)))
    print('avg test acc CNN: {}'.format(np.mean(cnn_test_accs)))
    N_classes = len(testset[1])
    print('random guessing accuracy: {}'.format(1/N_classes))
    # c=correct, r0=initial rotation
    # assume random rotations p(r0)=1/24
    # p(c) = p(c|r0)p(r0) + p(c|!r0)p(!r0)
    #      =    1    1/24 +  1/N_cl  23/24
    print('theoretically estimated CNN accuracy: {}'.format((1+23/N_classes)/24))


main()
