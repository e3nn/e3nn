# pylint: disable=C, R, not-callable, no-member, arguments-differ
from functools import partial

import torch

from se3cnn.non_linearities import GatedBlock
from se3cnn.point.operations import Convolution
from se3cnn.non_linearities import rescaled_act
from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import CosineBasisModel
from se3cnn.SO3 import rand_rot


def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L
    tetris = torch.tensor(tetris, dtype=torch.get_default_dtype())
    labels = torch.arange(len(tetris))

    # apply random rotation
    tetris = torch.stack([torch.einsum("ij,nj->ni", (rand_rot(), x)) for x in tetris])

    return tetris, labels


class AvgSpacial(torch.nn.Module):
    def forward(self, features):
        return features.flatten(2).mean(-1)


class SE3Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        features = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]
        self.num_features = len(features)

        sp = rescaled_act.Softplus(beta=5)

        RadialModel = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=50, act=sp)

        K = partial(Kernel, RadialModel=RadialModel)
        C = partial(Convolution, K)

        self.layers = torch.nn.ModuleList([
            GatedBlock(
                [(m, l) for l, m in enumerate(features[i])],
                [(m, l) for l, m in enumerate(features[i+1])],
                sp, rescaled_act.sigmoid, C)
            for i in range(len(features) - 1)
        ])
        self.layers += [AvgSpacial(), torch.nn.Linear(64, num_classes)]

    def forward(self, features, geometry):
        output = features
        for i in range(self.num_features - 1):
            output = self.layers[i](output, geometry)

        for i in range(self.num_features - 1, len(self.layers)):
            output = self.layers[i](output)

        return output


def main():
    torch.set_default_dtype(torch.float64)

    tetris, labels = get_dataset()
    tetris = tetris.cuda()
    labels = labels.cuda()
    f = SE3Net(len(tetris))
    f = f.cuda()

    optimizer = torch.optim.Adam(f.parameters())

    feature = tetris.new_ones(tetris.size(0), 1, tetris.size(1))

    for step in range(50):
        out = f(feature, tetris)
        loss = torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = out.argmax(1).eq(labels).double().mean().item()
        print("step={} loss={} accuracy={}".format(step, loss.item(), acc))

    out = f(feature, tetris)

    r_tetris, _ = get_dataset()
    r_tetris = r_tetris.cuda()
    r_out = f(feature, r_tetris)

    print('equivariance error={}'.format((out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
