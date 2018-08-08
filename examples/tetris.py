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


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


def low_pass_filter(image, scale):
    """
    :param image: [..., x, y, z]
    :param scale: float
    """
    if scale >= 1:
        return image

    dtype = image.dtype
    device = image.device

    sigma = 0.5 * (1 / scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=dtype, device=device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(size, 1, 1).expand(size, size, size)
    y = rng.view(1, size, 1).expand(size, size, size)
    z = rng.view(1, 1, size).expand(size, size, size)

    kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    out = F.conv3d(image.view(-1, 1, *image.size()[-3:]), kernel.view(1, 1, size, size, size), padding=size//2)
    out = out.view(*image.size())
    return out


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

    optimizer = torch.optim.Adam(network.parameters(), lr=3e-2, weight_decay=1e-5)
    for epoch in range(N_epochs):
        predictions = network(volumes)

        # decay learning rate
        optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, init_lr=1e-1, epoch_start=100, base_factor=.98, verbose=True)

        optimizer.zero_grad()
        F.cross_entropy(predictions, labels).backward()
        optimizer.step()

        argmax = predictions.argmax(1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('epoch {}: acc={}'.format(epoch, acc))


def test(network, dataset):
    network.eval()

    volumes, labels = dataset
    volumes = torch.tensor(volumes).cuda()
    labels = torch.tensor(labels).cuda()

    predictions = network(volumes)
    argmax = predictions.argmax(1)
    acc = (argmax.squeeze() == labels).float().mean().item()

    print('test acc={}'.format(acc))
    return acc


class SE3Net(torch.nn.Module):

    def __init__(self):
        super(SE3Net, self).__init__()
        features = [
            (1,),
            (2, 2, 2, 1),
            (4, 4, 4, 0),
            (6, 4, 4, 0),
            (64,)
        ]
        common_block_params = {
            'size': 5,
            'padding': 4,
            'dilation': 2,
            'activation': (F.relu, torch.sigmoid),
            'capsule_dropout_p': .0
        }

        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params) for i in range(len(features) - 1)]
        self.sequence = torch.nn.Sequential(*blocks,
                                            AvgSpacial(),
                                            nn.Dropout(p=.2),
                                            nn.Linear(64, 10))

    def forward(self, inp):  # pylint: disable=W
        inp = low_pass_filter(inp, 1 / 2)
        return self.sequence(inp)


# class CNN(torch.nn.Module):

#     def __init__(self):
#         super(CNN, self).__init__()

#         common_params = {
#             'kernel_size': 5,
#             'stride': 2,
#             'padding': 2,
#             'bias': True,
#         }

#         self.sequence = torch.nn.Sequential(
#             nn.Conv3d(1, 16, **common_params), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
#             nn.Conv3d(16, 32, **common_params), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
#             nn.Conv3d(32, 32, **common_params), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
#             nn.Conv3d(32, 16, **common_params), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
#             AvgSpacial(),
#             torch.nn.Linear(16, 10))

#     def forward(self, inp):  # pylint: disable=W
#         return self.sequence(inp)


def main():
    torch.backends.cudnn.benchmark = True

    N_epochs = 250
    N_test = 100
    trainset = get_volumes(rotate=True)  # train with randomly rotated pieces but only once

    network = SE3Net().cuda()
    train(network, trainset, N_epochs=N_epochs)
    se3_test_accs = []
    for _ in range(N_test):
        testset = get_volumes(rotate=True)
        acc = test(network, testset)
        se3_test_accs.append(acc)

    # network = CNN().cuda()
    # train(network, trainset, N_epochs=N_epochs)
    # cnn_test_accs = []
    # for _ in range(N_test):
    #     testset = get_volumes(rotate90=True)
    #     acc = test(network, testset)
    #     cnn_test_accs.append(acc)

    print('avg test acc SE3: {}'.format(np.mean(se3_test_accs)))
    # print('avg test acc CNN: {}'.format(np.mean(cnn_test_accs)))
    # N_classes = len(testset[1])
    # print('random guessing accuracy: {}'.format(1 / N_classes))
    # # c=correct, r0=initial rotation
    # # assume random rotations p(r0)=1/24
    # # p(c) = p(c|r0)p(r0) + p(c|!r0)p(!r0)
    # #      =    1    1/24 +  1/N_cl  23/24
    # print('theoretically estimated CNN accuracy: {}'.format((1 + 23 / N_classes) / 24))


main()
