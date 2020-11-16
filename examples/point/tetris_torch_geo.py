# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, arguments-differ, abstract-method
import time
from functools import partial

import torch
import torch_geometric as tg
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from e3nn.point.data_helpers import DataNeighbors
from e3nn import o3, rsh, rs
from e3nn.networks import MLNetwork, make_gated_block
from e3nn.non_linearities.rescaled_act import swish
from e3nn.tensor_product import GroupedWeightedTensorProduct
from e3nn.radial import GaussianRadialModel
from e3nn.linear import Linear


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


class Convolution(tg.nn.MessagePassing):
    """
    Convolution with self interaction
    """
    def __init__(self, Rs_in, Rs_out, lmax=3):
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

        Rs_sh = [(1, l, (-1)**l) for l in range(0, lmax + 1)]

        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.lin1 = Linear(Rs_in, Rs_out, allow_unused_inputs=True, allow_zero_outputs=True)
        self.tp = GroupedWeightedTensorProduct(Rs_in, Rs_sh, Rs_out, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.lin2 = Linear(Rs_out, Rs_out)
        self.Rs_sh = Rs_sh

    def forward(self, features, edge_index, edge_r, sh=None, size=None, n_norm=1):
        # features = [num_atoms, dim(Rs_in)]
        if sh is None:
            sh = rsh.spherical_harmonics_xyz(self.Rs_sh, edge_r, "component")  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_r.norm(dim=1))  # [num_messages, nweight]

        self_interation = self.lin1(features)
        features = self.propagate(edge_index, size=size, x=features, sh=sh, w=w)
        features = self.lin2(features)
        has_self_interaction = torch.cat([
            torch.ones(mul * (2 * l + 1)) if any(l_in == l and p_in == p for _, l_in, p_in in self.Rs_in) else torch.zeros(mul * (2 * l + 1))
            for mul, l, p in self.Rs_out
        ])
        return 0.5**0.5 * self_interation + (1 + (0.5**0.5 - 1) * has_self_interaction) * features

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)


def forward(f, shapes, labels, lmax, device):
    r_max = 1.1
    x = torch.ones(4, 1)
    batch = Batch.from_data_list([DataNeighbors(x, shape, r_max, y=label, self_interaction=False) for shape, label in zip(shapes, labels)])
    batch = batch.to(device)
    sh = rsh.spherical_harmonics_xyz(list(range(lmax + 1)), batch.edge_attr, 'component')
    out = f(batch.x, batch.edge_index, batch.edge_attr, sh=sh, n_norm=3)
    out = scatter_add(out, batch.batch, dim=0)
    out = torch.tanh(out)
    return out


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Define the network
    Rs_in = [(1, 0, 1)]
    Rs_out = [(1, 0, -1), (6, 0, 1)]
    lmax = 2

    f = MLNetwork(Rs_in, Rs_out, partial(Convolution, lmax=lmax), partial(make_gated_block, mul=16, lmax=lmax), layers=4)
    f = f.to(device)

    # Train the network on a tetris dataset
    tetris, labels = get_dataset()
    optimizer = torch.optim.Adam(f.parameters(), lr=3e-3)

    wall = time.perf_counter()
    for step in range(100):
        out = forward(f, tetris, labels, lmax, device)

        acc = out.cpu().round().eq(labels).double().mean().item()

        with torch.no_grad():
            r_out = forward(f, *get_dataset(), lmax, device)

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
