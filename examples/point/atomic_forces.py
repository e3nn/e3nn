# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name, abstract-method
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from e3nn.point.data_helpers import DataNeighbors
from e3nn.networks import MLNetwork, make_gated_block
from e3nn.non_linearities.rescaled_act import swish
from e3nn.point.message_passing import WTPConv
from e3nn.radial import GaussianRadialModel


def load_dataset(device='cpu'):
    # http://www.quantum-machine.org/gdml/data/npz/benzene_old_dft.npz
    data = np.load('benzene_old_dft.npz')

    geometry = torch.from_numpy(data['R'])
    geometry = geometry.to(device).contiguous()
    n = geometry.size(0)

    forces = torch.from_numpy(data['F'])
    forces = forces.to(device)

    # energies = torch.from_numpy(data['E'])
    # atoms_pattern = torch.from_numpy(data['z']).long()

    features = geometry.new_zeros(n, 12, 2)
    features[:, :6, 0] = 1
    features[:, 6:, 1] = 1
    features.mul_(2 ** 0.5)
    assert abs(features.pow(2).mean() - 1) < 1e-5
    features = features.to(device)

    return features, geometry, forces


def GatedNetwork(Rs_in, Rs_out, mul, lmax, layers=3, min_radius=0.0, max_radius=1.0, radial_basis=3, radial_neurons=100, radial_layers=2, radial_act=swish):
    def layer(Rs1, Rs2):
        R = partial(
            GaussianRadialModel,
            max_radius=max_radius,
            number_of_basis=radial_basis,
            h=radial_neurons,
            L=radial_layers,
            act=radial_act,
            min_radius=min_radius
        )
        Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]
        return WTPConv(Rs1, Rs2, Rs_sh, R)
    def activation(Rs):
        return make_gated_block(Rs, mul, lmax)
    return MLNetwork(Rs_in, Rs_out, layer, activation, layers)


def main():
    torch.set_default_dtype(torch.float64)

    Rs_in = [(2, 0, 1)]  # 2 (even-)scalars for one-hot encoding of atom type
    Rs_out = [(1, 0, 1)]  # predict the energy (even-scalar)

    r_max = 2
    f = GatedNetwork(Rs_in, Rs_out, mul=10, lmax=2, layers=2, max_radius=r_max)

    features, geometry, forces = load_dataset()

    i = torch.randperm(len(features))

    # trainset indices
    tr = i[:20]

    # testset indices
    te = i[20:50]

    optim = torch.optim.Adam(f.parameters(), lr=1e-2)

    def step(i):
        r = geometry[tr].detach().clone()
        r.requires_grad_(True)

        batch = Batch.from_data_list([DataNeighbors(x, r, r_max, self_interaction=False) for x, r in zip(features[tr], r)])

        out = f(batch.x, batch.edge_index, batch.edge_attr, n_norm=2)
        out = scatter_add(out, batch.batch, dim=0)

        pred_forces, = torch.autograd.grad(out.sum(), r, create_graph=True)

        loss = (pred_forces - forces[tr]).pow(2).sum(2).mean()
        print('loss {} = {}'.format(i, loss.item()))

        optim.zero_grad()
        loss.backward()
        optim.step()

    for i in range(50):
        step(i)

    r = geometry[te].detach().clone()
    r.requires_grad_(True)

    batch = Batch.from_data_list([DataNeighbors(x, r, r_max, self_interaction=False) for x, r in zip(features[te], r)])

    out = f(batch.x, batch.edge_index, batch.edge_attr, n_norm=2)
    out = scatter_add(out, batch.batch, dim=0)

    out.sum().backward()

    pred_forces = r.grad

    loss = (pred_forces - forces[te]).pow(2).sum(2).mean()
    print("test loss=", loss.item())


    plt.figure(figsize=(13, 13))
    for i in range(9):
        plt.subplot(331 + i)
        plt.plot(
            r[i, :, 0].detach().numpy(),
            r[i, :, 1].detach().numpy(),
            '.k',
            markersize=10,
            zorder=-2,
        )
        plt.quiver(
            r[i, :, 0].detach().numpy(),
            r[i, :, 1].detach().numpy(),
            pred_forces[i, :, 0].numpy(),
            pred_forces[i, :, 1].numpy()
        )
        plt.quiver(
            r[i, :, 0].detach().numpy(),
            r[i, :, 1].detach().numpy(),
            forces[te][i, :, 0].numpy(),
            forces[te][i, :, 1].numpy(),
            color='red'
        )

        plt.legend(['atoms', 'prediction', 'true forces'])

        plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('atomic_forces.jpeg')


if __name__ == '__main__':
    main()
