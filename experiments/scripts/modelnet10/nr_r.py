# pylint: disable=C,R,E1101,E1102,W0622
import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from se3cnn import basis_kernels
from functools import partial
from se3cnn.blocks import GatedBlock
from experiments.datasets.modelnet.modelnet_old import ModelNet10, Obj2Voxel, CacheNPY


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.int_repr = None

        features = [
            (1, ),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (100, )
        ]

        radial_window = partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                mode='compromise', border_dist=0, sigma=0.6)

        common_block_params = {
            'size': 5,
            'stride': 2,
            'padding': 3,
            'normalization': 'batch',
            'radial_window': radial_window
        }

        block_params = [
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
            nn.Linear(features[-1][0], 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''

        self.int_repr = []

        for m in self.sequence:
            x = m(x)
            # print(x.size())
            self.int_repr.append(x)
        # x = self.sequence(x)  # [batch, features]

        return x


def train(model, dataset, n_epoch):
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    data = []

    for epoch in range(n_epoch):
        for i, (input, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # forward and backward propagation
            output = model(input)
            loss = F.nll_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # download results on the CPU
            loss = loss.detach().cpu().item()
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            # compute the accuracy
            acc = float(np.sum(output.argmax(-1) == target) / target.size)

            data.append({
                "epoch": epoch,
                "i": i / len(dataloader),
                "loss": loss,
                "accuracy": acc
            })

            print("{}:{}/{}: acc={}% loss={}".format(epoch, i, len(dataloader), 100 * acc, loss))
    return data


def test(model, dataset):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

    correct = 0
    for i, (input, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # forward and backward propagation
        output = model(input)

        # download results on the CPU
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # compute the accuracy
        correct += np.sum(output.argmax(-1) == target)

        print("{}/{}".format(i, len(dataloader)))

    return correct / len(dataset)


def main():

    torch.backends.cudnn.benchmark = True

    def compose(t1, t2):
        def f(x):
            return t2(t1(x))
        return f

    def to_tensor(x):
        return torch.tensor(x.astype(np.uint8), dtype=torch.float).unsqueeze(0) / 8

    classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

    def target_transform(x):
        return classes.index(x)

    cache = CacheNPY("r32dd", repeat=1, transform=Obj2Voxel(32, rotate=True, double=True, diagonal_bounding_box=True))
    trainR = ModelNet10("./root/", mode='train',
                        classes=classes,
                        download=True,
                        transform=compose(cache, to_tensor),
                        target_transform=target_transform)

    cache = CacheNPY("nr32dd", repeat=1, transform=Obj2Voxel(32, rotate=False, double=True, diagonal_bounding_box=True))
    trainNR = ModelNet10("./root/", mode='train',
                         classes=classes,
                         download=True,
                         transform=compose(cache, to_tensor),
                         target_transform=target_transform)

    cache = CacheNPY("r32dd", repeat=1, transform=Obj2Voxel(32, rotate=True, double=True, diagonal_bounding_box=True))
    testR = ModelNet10("./root/", mode='validation',
                       classes=classes,
                       download=True,
                       transform=compose(cache, to_tensor),
                       target_transform=target_transform)

    cache = CacheNPY("nr32dd", repeat=1, transform=Obj2Voxel(32, rotate=False, double=True, diagonal_bounding_box=True))
    testNR = ModelNet10("./root/", mode='validation',
                        classes=classes,
                        download=True,
                        transform=compose(cache, to_tensor),
                        target_transform=target_transform)

    results = []

    for trainset in [trainNR, trainR]:
        model = CNN()
        if torch.cuda.is_available():
            model.cuda()

        train(model, trainset, 100)

        for testset in [testNR, testR]:
            results.append((test(model, trainset), test(model, testset)))

    print(results)


main()
