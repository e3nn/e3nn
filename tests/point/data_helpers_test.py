# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
import e3nn.point.data_helpers as dh
from e3nn import rs


def test_data_helpers():
    N = 7
    lattice = torch.randn(3, 3)
    pos = torch.randn(N, 3)
    Rs_in = [(3, 0), (1, 1)]
    x = torch.randn(N, rs.dim(Rs_in))
    r_max = 1
    dh.neighbor_list_and_relative_vec_lattice(pos, lattice, r_max)
    dh.DataPeriodicNeighbors(x, Rs_in, pos, lattice, r_max)
    dh.neighbor_list_and_relative_vec(pos, r_max)
    dh.DataNeighbors(x, Rs_in, pos, r_max)


def test_silicon_neighbors():
    lattice = torch.tensor([
       [3.34939851, 0.        , 1.93377613],
       [1.11646617, 3.1578432 , 1.93377613],
       [0.        , 0.        , 3.86755226]
    ])
    coords = torch.tensor([
        [0.        , 0.        , 0.        ],
        [1.11646617, 0.7894608 , 1.93377613]
    ])
    species = ['Si', 'Si']
    r_max = 2.5
    edge_index, edge_attr = dh.neighbor_list_and_relative_vec_lattice(coords, lattice, r_max=r_max)
    edge_index_true = torch.LongTensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    ])
    torch.allclose(edge_index, edge_index_true)
