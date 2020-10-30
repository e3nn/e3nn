# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, arguments-differ, abstract-method
from functools import partial

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

import e3nn.point.data_helpers as dh
from e3nn import o3
from e3nn.kernel import Kernel
from e3nn.networks import GatedConvNetwork
from e3nn.point.message_passing import Convolution


def convolution(Rs_in, Rs_out, lmax, RadialModel):
    return Convolution(Kernel(Rs_in, Rs_out, RadialModel, partial(o3.selection_rule_in_out_sh, lmax=lmax)))


def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L
    tetris = torch.tensor(tetris, dtype=torch.get_default_dtype())
    labels = torch.arange(len(tetris))

    # apply random rotation
    tetris = torch.einsum('ij,zaj->zai', o3.rand_rot(), tetris)

    return tetris, labels


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x = torch.ones(4, 1)
    Rs_in = [0]
    r_max = 1.1

    tetris, labels = get_dataset()
    tetris_dataset = [dh.DataNeighbors(x, Rs_in, shape, r_max, y=label) for shape, label in zip(tetris, labels)]

    Rs_hidden = [(16, 0), (16, 1), (16, 2)]
    Rs_out = [(len(tetris), 0)]
    lmax = 3

    f = GatedConvNetwork(Rs_in, Rs_hidden, Rs_out, lmax, convolution=convolution)
    f = f.to(device)

    batch = Batch.from_data_list(tetris_dataset)
    batch = batch.to(device)

    optimizer = torch.optim.Adam(f.parameters(), lr=3e-3)

    for step in range(50):
        out = f(batch.x, batch.edge_index, batch.edge_attr)
        out = scatter_add(out, batch.batch, dim=0)

        acc = out.cpu().argmax(1).eq(labels).double().mean().item()

        r_tetris_dataset = [dh.DataNeighbors(x, Rs_in, shape, r_max, y=label) for shape, label in zip(*get_dataset())]
        r_batch = Batch.from_data_list(r_tetris_dataset)
        r_batch = r_batch.to(device)

        with torch.no_grad():
            r_out = f(r_batch.x, r_batch.edge_index, r_batch.edge_attr)
            r_out = scatter_add(r_out, r_batch.batch, dim=0)

        loss = torch.nn.functional.cross_entropy(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step={} loss={:.2e} accuracy={:.2f} equivariance error={:.1e}".format(step, loss.item(), acc, (out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
