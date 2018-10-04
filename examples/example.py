# pylint: disable=C,R,E1101,E1102
'''
Minimalist example of usage of SE(3) CNN

This example train a neural network to classify two geometrical objects
A ball : x^2 + y^2 + z^2 < radius^2
An octahedron : |x| + |y| + |z| < radius

The input data is generated on the fly, the radius is random and noise is added
'''
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# The class GatedBlock inherit from the class torch.nn.Module.
# It contains one convolution, some ReLU and multiplications
from se3cnn.blocks import GatedBlock


class AvgSpacial(nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # The parameters of a GatedBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - the non linearities for the scalars and the gates (None for no non-linearity)
        # - stride, padding... same as 2D convolution
        features = [
            (1, ),  # As input we have a scalar field
            (2, 2, 2, 2),  # Note that this particular choice of multiplicities it completely arbitrary
            (4, 4, 3, 3),
            (4, 4, 3, 3),
            (4, 4, 3, 3),
            (4, 4, 3, 3),
            (4, 4, 3, 3),
            (20, )  # scalar fields to end with fully-connected layers
        ]

        common_block_params = {
            'size': 5,
            'stride': 2,
            'padding': 3,
        }

        block_params = [
            {'activation': (None, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]

        self.sequence = nn.Sequential(
            *blocks,
            AvgSpacial(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''

        x = self.sequence(inp)  # [batch, features]

        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    model = CNN()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch_size = 64
    sample_size = 24  # Size of the input cube

    mesh = np.linspace(-1, 1, sample_size)
    mx, my, mz = np.meshgrid(mesh, mesh, mesh)

    def step(i):
        # only noise
        x = 0.1 * np.random.randn(batch_size, 1, sample_size, sample_size, sample_size)
        # random labels: zero for the ball and one for the octahedron
        y = np.random.randint(0, 2, size=(batch_size,))

        # add the balls and octahedrons on top of the noise
        for j, label in enumerate(y):
            radius = 0.6 + np.random.rand() * (0.9 - 0.6)

            if label == 0:
                # ball
                mask = mx ** 2 + my ** 2 + mz ** 2 < radius ** 2 / np.pi ** (2 / 3)
            if label == 1:
                # octahedron
                mask = abs(mx) + abs(my) + abs(mz) < radius

            x[j, 0, mask] += np.random.randint(2) * 2 - 1

        # convert the input and labels into Pytorch tensor
        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        # forward and backward propagation
        out = model(x)
        loss = F.nll_loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute the accuracy
        acc = out.argmax(1).eq(y).long().sum().item() / batch_size

        print("{:d}: acc={:.1f}% loss={:.2e}".format(i, 100 * acc, loss.item()))

    for i in range(1000):
        step(i)


if __name__ == '__main__':
    main()
