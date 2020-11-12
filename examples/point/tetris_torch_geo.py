# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, arguments-differ, abstract-method
import time
from functools import partial

import torch
import torch_geometric as tg
from torch_geometric.data import Batch
from torch_scatter import scatter_add

import e3nn.point.data_helpers as dh
from e3nn import o3, rsh
from e3nn.networks import MLNetwork, make_gated_block
from e3nn.non_linearities.rescaled_act import swish
from e3nn.tensor_product import GroupedWeightedTensorProduct
from e3nn.radial import GaussianRadialModel
from e3nn.linear import Linear


class Convolution(tg.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out):
        super().__init__(aggr='add', flow='target_to_source')
        RadialModel = partial(
            GaussianRadialModel,
            max_radius=1.2,
            min_radius=0.0,
            number_of_basis=3,
            h=100,
            L=2,
            act=swish
        )

        Rs_sh = [(1, l, (-1)**l) for l in range(0, 3 + 1)]

        self.tp = GroupedWeightedTensorProduct(Rs_in, Rs_sh, Rs_out, groups=4, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.lin = Linear(Rs_out, Rs_out)
        self.Rs_sh = Rs_sh

    def forward(self, features, edge_index, edge_r, sh=None, size=None, n_norm=1):
        if sh is None:
            sh = rsh.spherical_harmonics_xyz(self.Rs_sh, edge_r, "component")  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_r.norm(dim=1))  # [num_messages, nweight]

        features = self.propagate(edge_index, size=size, x=features, sh=sh, w=w)
        features = self.lin(features)
        return features

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)


def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # zigzag
    tetris = torch.tensor(tetris, dtype=torch.get_default_dtype())
    labels = torch.tensor([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=torch.get_default_dtype())

    # apply random rotation
    tetris = torch.einsum('ij,zaj->zai', o3.rand_rot(), tetris)

    return tetris, labels


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x = torch.ones(4, 1)
    Rs_in = [(1, 0, 1)]
    r_max = 1.1

    tetris, labels = get_dataset()
    tetris_dataset = [dh.DataNeighbors(x, shape, r_max, y=label) for shape, label in zip(tetris, labels)]

    Rs_out = [(1, 0, -1), (6, 0, 1)]
    lmax = 3

    f = MLNetwork(Rs_in, Rs_out, Convolution, partial(make_gated_block, mul=16, lmax=lmax), 2)
    f = f.to(device)

    batch = Batch.from_data_list(tetris_dataset)
    batch = batch.to(device)
    sh = rsh.spherical_harmonics_xyz(list(range(lmax + 1)), batch.edge_attr, 'component')

    optimizer = torch.optim.Adam(f.parameters(), lr=3e-3)

    wall = time.perf_counter()
    for step in range(100):
        out = f(batch.x, batch.edge_index, batch.edge_attr, sh=sh, n_norm=3)
        out = scatter_add(out, batch.batch, dim=0)
        out = torch.tanh(out)

        acc = out.cpu().round().eq(labels).double().mean().item()

        r_tetris_dataset = [dh.DataNeighbors(x, shape, r_max, y=label) for shape, label in zip(*get_dataset())]
        r_batch = Batch.from_data_list(r_tetris_dataset)
        r_batch = r_batch.to(device)
        r_sh = rsh.spherical_harmonics_xyz(list(range(lmax + 1)), r_batch.edge_attr, 'component')

        with torch.no_grad():
            r_out = f(r_batch.x, r_batch.edge_index, r_batch.edge_attr, sh=r_sh, n_norm=3)
            r_out = scatter_add(r_out, r_batch.batch, dim=0)
            r_out = torch.tanh(r_out)

        loss = (out - labels).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("wall={:.1f} step={} loss={:.2e} accuracy={:.2f} equivariance error={:.1e}".format(
            time.perf_counter() - wall, step, loss.item(), acc, (out - r_out).pow(2).mean().sqrt().item()))

    print(labels.numpy().round(1))
    print(out.detach().numpy().round(1))


if __name__ == '__main__':
    main()
