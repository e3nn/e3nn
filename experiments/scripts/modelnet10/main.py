# pylint: disable=C,R,E1101,W0622
import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from se3_cnn import basis_kernels
from functools import partial
from se3_cnn.blocks import GatedBlock
from experiments.datasets.modelnet.modelnet import ModelNet10, Obj2Voxel, CacheNPY


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        features = [
            (1, ),
            (2, 2, 2),
            (4, 4, 3),
            (10, )
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
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''

        x = self.sequence(inp)  # [batch, features]

        return x


def main():
    torch.backends.cudnn.benchmark = True

    model = CNN()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    cache = CacheNPY("v64", repeat=24, transform=Obj2Voxel(64))

    def transform(x):
        x = cache(x)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def target_transform(x):
        classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
        return classes.index(x)

    trainset = ModelNet10("./root/", train=True, download=True, transform=transform, target_transform=target_transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, drop_last=True)

    for epoch in range(10):
        for input, target in dataloader:
            input = Variable(input)
            target = Variable(target)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # forward and backward propagation
            output = model(input)
            optimizer.zero_grad()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # download results on the CPU
            loss = loss.data.cpu().numpy()
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            # compute the accuracy
            acc = np.sum(output.argmax(-1) == target) / output.size

            print("{}: acc={}% loss={}".format(epoch, 100 * acc, float(loss)))


if __name__ == '__main__':
    main()
