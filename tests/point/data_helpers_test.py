# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
import e3nn.point.data_helpers as dh
from e3nn import rs
import numpy as np

torch.set_default_dtype(torch.float64)


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
    r_max = 2.5
    edge_index, edge_attr = dh.neighbor_list_and_relative_vec_lattice(coords, lattice, r_max=r_max)
    edge_index_true = torch.LongTensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    ])
    torch.allclose(edge_index, edge_index_true)


def test_get_edge_edges_and_index():
    edge_index = torch.LongTensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ])
    edge_index_dict_asym, _, edge_edge_index_asym = dh.get_edge_edges_and_index(edge_index, symmetric_edges=False)
    edge_index_dict_symm, _, edge_edge_index_symm = dh.get_edge_edges_and_index(edge_index, symmetric_edges=True)

    check1 = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}
    check2 = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}

    assert edge_index_dict_asym == check1
    assert edge_index_dict_symm == check2
    assert np.max(list(edge_index_dict_asym.values())) == np.max(edge_edge_index_asym)
    assert np.max(list(edge_index_dict_symm.values())) == np.max(edge_edge_index_symm)


def test_initialize_edges():
    edge_index = torch.LongTensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    edge_index_dict, _, _ = dh.get_edge_edges_and_index(edge_index, symmetric_edges=True)
    _, Rs = dh.initialize_edges(torch.ones(5, 1), [(1, 0, 1)], torch.randn(5, 3), edge_index_dict, 2, symmetric_edges=True)
    assert Rs == [(1, 0, 1), (1, 1, -1), (1, 2, 1)]

    _, Rs = dh.initialize_edges(torch.ones(5, 3), [(1, 1, -1)], torch.randn(5, 3), edge_index_dict, 0, symmetric_edges=True)
    assert Rs == [(1, 0, 1), (1, 2, 1)]

    edge_index_dict, _, _ = dh.get_edge_edges_and_index(edge_index, symmetric_edges=False)
    _, Rs = dh.initialize_edges(torch.ones(5, 3), [(1, 1, -1)], torch.randn(5, 3), edge_index_dict, 0, symmetric_edges=False)
    assert Rs == [(1, 0, 1), (1, 1, 1), (1, 2, 1)]


def test_DataEdgeNeighbors():
    square = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
    )
    square -= square.mean(-2)
    data = dh.DataEdgeNeighbors(torch.ones(4, 1), [(1, 0, 1)], square, 1.5, 2)
    assert list(data.edge_x.shape) == [16, 9]
    assert list(data.edge_edge_index.shape) == [2, 64]
    assert list(data.edge_edge_attr.shape) == [64, 3]


def test_DataEdgePeriodicNeighbors():
    pos = torch.ones(1, 3) * 0.5
    lattice = torch.eye(3)
    dh.DataEdgePeriodicNeighbors(torch.ones(1, 1), [(1, 0, 1)], pos, lattice, 1.5, 2)
