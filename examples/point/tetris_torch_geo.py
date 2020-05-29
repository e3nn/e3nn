# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, arguments-differ
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

import e3nn.point.data_helpers as dh
from e3nn.networks import GatedConvNetwork
from e3nn.o3 import rand_rot
from e3nn.point.message_passing import Convolution


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
    tetris = torch.einsum('ij,zaj->zai', rand_rot(), tetris)

    return tetris, labels


class SumNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, *args, batch=None, **kwargs):
        output = self.network(*args, **kwargs)
        return scatter_add(output, batch, dim=0)


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tetris, labels = get_dataset()

    x = torch.ones(4, 1)
    Rs_in = [(1, 0)]
    r_max = 1.1
    tetris_dataset = []
    for shape, label in zip(tetris, labels):
        data = dh.DataNeighbors(x, Rs_in, shape, r_max, y=torch.tensor([label]))
        tetris_dataset.append(data)

    Rs_hidden = [(16, 0), (16, 1), (16, 2)]
    Rs_out = [(len(tetris), 0)]
    lmax = 3

    f = SumNetwork(Rs_in, Rs_hidden, Rs_out, lmax, convolution=Convolution)
    f = f.to(device)

    batch = Batch.from_data_list(tetris_dataset)
    batch = batch.to(device)

    optimizer = torch.optim.Adam(f.parameters(), lr=3e-3)

    for step in range(50):
        N, _ = batch.x.shape
        out = f(batch.x, batch.edge_index, batch.edge_attr, size=N, batch=batch.batch)
        loss = torch.nn.functional.cross_entropy(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = out.cpu().argmax(1).eq(labels).double().mean().item()

        out = f(batch.x, batch.edge_index, batch.edge_attr, size=N, batch=batch.batch)

        r_tetris, _ = get_dataset()

        r_tetris_dataset = []
        for shape, label in zip(r_tetris, labels):
            data = dh.DataNeighbors(x, Rs_in, shape, r_max, y=torch.tensor([label]))
            r_tetris_dataset.append(data)

        r_batch = Batch.from_data_list(r_tetris_dataset)
        r_batch = r_batch.to(device)

        r_out = f(r_batch.x, r_batch.edge_index, r_batch.edge_attr, size=N, batch=r_batch.batch)

        print("step={} loss={:.2e} accuracy={:.2f} equivariance error={:.1e}".format(step, loss.item(), acc, (out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
