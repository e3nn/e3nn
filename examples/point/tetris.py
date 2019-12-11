# pylint: disable=C, R, not-callable, no-member, arguments-differ
from functools import partial

import torch

from e3nn.non_linearities import GatedBlock
from e3nn.point.operations import Convolution
from e3nn.non_linearities.rescaled_act import relu, sigmoid
from e3nn.point.kernel import Kernel
from e3nn.point.radial import CosineBasisModel
from e3nn.SO3 import rand_rot


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
        return features.mean(1)


class SE3Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        representations = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]
        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]

        R = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=50, act=relu)
        K = partial(Kernel, RadialModel=R)
        C = partial(Convolution, K)

        self.firstlayers = torch.nn.ModuleList([
            GatedBlock(Rs_in, Rs_out, relu, sigmoid, C)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])
        self.lastlayers = torch.nn.Sequential(AvgSpacial(), torch.nn.Linear(64, num_classes))

    def forward(self, features, geometry):
        for m in self.firstlayers:
            features = m(features.div(4 ** 0.5), geometry)

        return self.lastlayers(features)


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tetris, labels = get_dataset()
    tetris = tetris.to(device)
    labels = labels.to(device)
    f = SE3Net(len(tetris))
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters())

    feature = tetris.new_ones(tetris.size(0), tetris.size(1), 1)

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
    r_tetris = r_tetris.to(device)
    r_out = f(feature, r_tetris)

    print('equivariance error={}'.format((out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
