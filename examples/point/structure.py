# pylint: disable=C, R, not-callable, no-member, arguments-differ
import json
from functools import partial

import pymatgen
import torch
import random
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import relu, sigmoid
from e3nn.kernel import Kernel
from e3nn.point.operations import PeriodicConvolution
from e3nn.radial import CosineBasisModel


def get_dataset(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)

    structures = [pymatgen.Structure.from_dict(s) for s, l in dataset]
    classes = ['diamond', 'fcc', 'bcc', 'hcp', 'rutile', 'perovskite', 'spinel', 'corundum']
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

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, relu, sigmoid)
            conv = PeriodicConvolution(K, Rs_in, act.Rs_in, max_radius=3.8)
            return torch.nn.ModuleList([conv, act])

        self.firstlayers = torch.nn.ModuleList([
            make_layer(Rs_in, Rs_out)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])
        self.lastlayers = torch.nn.Sequential(AvgSpacial(), torch.nn.Linear(64, num_classes))

    def forward(self, structure):
        p = next(self.parameters())
        geometry = torch.stack([p.new_tensor(s.coords) for s in structure.sites])
        features = p.new_ones(1, len(geometry), 1)
        geometry = geometry.unsqueeze(0)

        for i, (conv, act) in enumerate(self.firstlayers):
            assert torch.isfinite(features).all(), i
            features = conv(features, geometry, structure.lattice, n_norm=4)
            features = act(features)

        return self.lastlayers(features).squeeze(0)


def main():
    import time
    torch.manual_seed(42)
    random.seed(42)
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    structures, labels = get_dataset('structure-1atomstype-trainset.json')
    labels = torch.tensor(labels, device=device)

    f = Network(8)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters())
    success = []

    t1 = time.time()
    for step in range(800):
        i = random.randint(0, len(structures) - 1)
        struct = structures[i]
        target = labels[i]

        out = f(struct)
        success.append(1 if out.argmax().item() == target else 0)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), target.unsqueeze(0))
        loss.backward()

        if step % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
            # print("step={} loss={:.2e} {}".format(step, loss.item(), success[-10:]))

    t2 = time.time()
    print(f"Training time: {t2-t1:.2f} seconds")

    def test(filename):
        structures, labels = get_dataset(filename)
        pred = [f(s).argmax().item() for s in structures]
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(labels, pred))

    with torch.no_grad():
        test('structure-1atomstype-trainset.json')
        test('structure-1atomstype-testset.json')


if __name__ == '__main__':
    main()
