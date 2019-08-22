# pylint: disable=C, R, not-callable, no-member, arguments-differ
import json
from functools import partial

import pymatgen
import torch
import random
from se3cnn.non_linearities import GatedBlock
from se3cnn.non_linearities.rescaled_act import relu, sigmoid
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import PeriodicConvolution
from se3cnn.point.radial import CosineBasisModel


def get_dataset():
    with open('structure-dataset.json', 'r') as f:
        dataset = json.load(f)

    structures = [pymatgen.Structure.from_dict(s) for s, l in dataset]
    classes = sorted({l for s, l in dataset})
    labels = [classes.index(l) for s, l in dataset]

    return structures, labels


class AvgSpacial(torch.nn.Module):
    def forward(self, features):
        return features.mean(1)


class Network(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        representations = [(1,), (4, 4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4), (64,)]
        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]

        R = partial(CosineBasisModel, max_radius=3.8, number_of_basis=10, h=100, L=2, act=relu)
        K = partial(Kernel, RadialModel=R)
        C = partial(PeriodicConvolution, K)

        self.firstlayers = torch.nn.ModuleList([
            GatedBlock(Rs_in, Rs_out, relu, sigmoid, C)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])
        self.lastlayers = torch.nn.Sequential(AvgSpacial(), torch.nn.Linear(64, num_classes))

    def forward(self, features, geometry, lattice):
        for m in self.firstlayers:
            features = m(features.div(4 ** 0.5), geometry, lattice, 3.8)

        return self.lastlayers(features)


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    structures, labels = get_dataset()
    labels = torch.tensor(labels, device=device)

    f = Network(4)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters())
    success = []

    for step in range(500):
        i = random.randint(0, len(structures) - 1)
        struct = structures[i]
        target = labels[i]

        geo = torch.stack([torch.tensor(s.coords) for s in struct.sites])
        feature = geo.new_ones(len(geo), 1)

        out = f(feature.unsqueeze(0), geo.unsqueeze(0), struct.lattice)
        success.append(1 if out.argmax(1)[0].item() == target else 0)
        loss = torch.nn.functional.cross_entropy(out, target.unsqueeze(0))
        loss.backward()

        if step % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()

            print("step={} loss={:.2e} {}".format(step, loss.item(), success[-10:]))


if __name__ == '__main__':
    main()
