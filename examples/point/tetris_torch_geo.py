# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, arguments-differ, abstract-method
import time
from functools import partial

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from e3nn.point.data_helpers import DataNeighbors
from e3nn import o3, rsh
from e3nn.networks import MLNetwork, make_gated_block
from e3nn.non_linearities.rescaled_act import swish
from e3nn.point.message_passing import WTPConv2
from e3nn.radial import GaussianRadialModel


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


def forward(f, shapes, Rs_sh, lmax, device):
    r_max = 1.1
    x = torch.ones(4, 1)
    batch = Batch.from_data_list([DataNeighbors(x, shape, r_max, self_interaction=False) for shape in shapes])
    batch = batch.to(device)
    # Pre-compute the spherical harmonics and re-use them in each convolution
    sh = rsh.spherical_harmonics_xyz(Rs_sh, batch.edge_attr, 'component')
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
    Rs_sh = [(1, l, (-1)**l) for l in range(lmax+1)]
    RadialModel = partial(
        GaussianRadialModel,
        max_radius=1.2,
        min_radius=0.0,
        number_of_basis=3,
        h=100,
        L=2,
        act=swish
    )

    f = MLNetwork(Rs_in, Rs_out, partial(WTPConv2, Rs_sh=Rs_sh, RadialModel=RadialModel), partial(make_gated_block, mul=16, lmax=lmax), layers=4)
    f = f.to(device)

    # Train the network on a tetris dataset
    tetris, labels = get_dataset()
    optimizer = torch.optim.Adam(f.parameters(), lr=3e-3)

    wall = time.perf_counter()
    for step in range(100):
        out = forward(f, tetris, Rs_sh, lmax, device)

        acc = out.cpu().round().eq(labels).double().mean().item()

        with torch.no_grad():
            shapes, _ = get_dataset()
            r_out = forward(f, shapes, Rs_sh, lmax, device)

        loss = (out - labels.to(device=out.device)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("wall={:.1f} step={} loss={:.2e} accuracy={:.2f} equivariance error={:.1e}".format(
            time.perf_counter() - wall, step, loss.item(), acc, (out - r_out).pow(2).mean().sqrt().item()))

    print(labels.numpy().round(1))
    print(out.detach().cpu().numpy().round(1))


if __name__ == '__main__':
    main()
