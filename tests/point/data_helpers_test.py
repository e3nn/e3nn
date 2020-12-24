# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, protected-access
import numpy as np
import torch
import e3nn.point.data_helpers as dh
from e3nn import rs

torch.set_default_dtype(torch.float64)


def edge_index_set_equiv(a, b):
    """Compare edge_index arrays in an unordered way."""
    # [[0, 1], [1, 0]] -> {(0, 1), (1, 0)}
    a = a.numpy()  # numpy gives ints when iterated, tensor gives non-identical scalar tensors.
    b = b.numpy()
    return set(zip(a[0], a[1])) == set(zip(b[0], b[1]))


def test_data_helpers():
    N = 7
    lattice = torch.randn(3, 3)
    pos = torch.randn(N, 3)
    Rs_in = [(3, 0), (1, 1)]
    x = torch.randn(N, rs.dim(Rs_in))
    r_max = 1
    dh._neighbor_list_and_relative_vec_lattice(pos, lattice, r_max)
    dh.DataPeriodicNeighbors(x, pos, lattice, r_max)
    dh._neighbor_list_and_relative_vec(pos, r_max)
    dh.DataNeighbors(x, pos, r_max)


def test_from_ase():
    import ase.build
    # Non-periodic
    atoms = ase.build.molecule('CH3CHO')
    data = dh.DataNeighbors.from_ase(atoms, r_max=2.)
    assert data.x.shape == (len(atoms), 3)  # 3 species in this atoms
    # check edges
    for edge in range(data.num_edges):
        real_displacement = atoms.positions[data.edge_index[1, edge]] - atoms.positions[data.edge_index[0, edge]]
        assert torch.allclose(data.edge_attr[edge], torch.as_tensor(real_displacement))
    # periodic
    atoms = ase.build.bulk('Cu', 'fcc', a=3.6, cubic=True)
    data = dh.DataNeighbors.from_ase(atoms, r_max=2.5)
    assert data.x.shape == (len(atoms), 1)  # one species


def test_positions_grad():
    N = 7
    lattice = torch.randn(3, 3)
    pos = torch.randn(N, 3)
    pos.requires_grad_(True)
    Rs_in = [(3, 0), (1, 1)]
    x = torch.randn(N, rs.dim(Rs_in))
    r_max = 1
    data = dh.DataNeighbors(x, pos, r_max)
    assert pos.requires_grad
    assert data.edge_attr.requires_grad
    torch.autograd.grad(data.edge_attr.sum(), pos, create_graph = True)
    data = dh.DataPeriodicNeighbors(x, pos, lattice, r_max)
    assert pos.requires_grad
    assert data.edge_attr.requires_grad
    torch.autograd.grad(data.edge_attr.sum(), pos, create_graph = True)


def test_some_periodic():
    import ase.build
    # monolayer in xy,
    atoms = ase.build.fcc111('Al', size=(3, 3, 1), vacuum=0.0)
    data = dh.DataNeighbors.from_ase(atoms, r_max=2.9)  # first shell dist is 2.864A
    # Check number of neighbors:
    _, neighbor_count = np.unique(data.edge_index[0].numpy(), return_counts=True)
    assert (neighbor_count == 7).all()  # 6 neighbors + self interaction
    # Check not periodic in z
    assert torch.allclose(data.edge_attr[:, 2], torch.zeros(data.num_edges))


def test_relative_vecs():
    coords = torch.tensor([
        [0, 0, 0],
        [1.11646617, 0.7894608, 1.93377613]
    ])
    r_max = 2.5
    data = dh.DataNeighbors(
        x=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        r_max=r_max,
    )
    edge_index_true = torch.LongTensor([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])
    assert edge_index_set_equiv(data.edge_index, edge_index_true)
    assert torch.allclose(
        coords[1] - coords[0],
        data.edge_attr[
            (data.edge_index[0] == 0) & (data.edge_index[1] == 1)
        ][0]
    )
    assert torch.allclose(
        coords[0] - coords[1],
        data.edge_attr[
            (data.edge_index[0] == 1) & (data.edge_index[1] == 0)
        ][0]
    )


def test_self_interaction():
    coords = torch.tensor([
        [0, 0, 0],
        [1.11646617, 0.7894608, 1.93377613]
    ])
    r_max = 2.5
    data_no_si = dh.DataNeighbors(
        x=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        r_max=r_max,
        self_interaction=False,
    )
    true_no_si = torch.LongTensor([
        [0, 1],
        [1, 0]
    ])
    assert edge_index_set_equiv(data_no_si.edge_index, true_no_si)
    data_si = dh.DataNeighbors(
        x=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        r_max=r_max,
        self_interaction=True
    )
    true_si = torch.LongTensor([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])
    assert edge_index_set_equiv(data_si.edge_index, true_si)


def test_silicon_neighbors():
    lattice = torch.tensor([
        [3.34939851, 0, 1.93377613],
        [1.11646617, 3.1578432, 1.93377613],
        [0, 0, 3.86755226]
    ])
    coords = torch.tensor([
        [0, 0, 0],
        [1.11646617, 0.7894608, 1.93377613]
    ])
    r_max = 2.5
    edge_index, _edge_attr = dh._neighbor_list_and_relative_vec_lattice(coords, lattice, r_max=r_max)
    edge_index_true = torch.LongTensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    ])
    assert edge_index_set_equiv(edge_index, edge_index_true)
    data = dh.DataNeighbors(
        x=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        r_max=r_max,
        cell=lattice,
        pbc=True,
    )
    assert edge_index_set_equiv(data.edge_index, edge_index_true)


def test_get_edge_edges_and_index():
    edge_index = torch.LongTensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ])
    edge_index_dict_asym, _, edge_edge_index_asym = dh._get_edge_edges_and_index(edge_index, symmetric_edges=False)
    edge_index_dict_symm, _, edge_edge_index_symm = dh._get_edge_edges_and_index(edge_index, symmetric_edges=True)

    check1 = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}
    check2 = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}

    assert edge_index_dict_asym == check1
    assert edge_index_dict_symm == check2
    assert np.max(list(edge_index_dict_asym.values())) == np.max(edge_edge_index_asym)
    assert np.max(list(edge_index_dict_symm.values())) == np.max(edge_edge_index_symm)


def test_initialize_edges():
    edge_index = torch.LongTensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    edge_index_dict, _, _ = dh._get_edge_edges_and_index(edge_index, symmetric_edges=True)
    _, Rs = dh._initialize_edges(torch.ones(5, 1), [(1, 0, 1)], torch.randn(5, 3), edge_index_dict, 2, symmetric_edges=True)
    assert Rs == [(1, 0, 1), (1, 1, -1), (1, 2, 1)]

    _, Rs = dh._initialize_edges(torch.ones(5, 3), [(1, 1, -1)], torch.randn(5, 3), edge_index_dict, 0, symmetric_edges=True)
    assert Rs == [(1, 0, 1), (1, 2, 1)]

    edge_index_dict, _, _ = dh._get_edge_edges_and_index(edge_index, symmetric_edges=False)
    _, Rs = dh._initialize_edges(torch.ones(5, 3), [(1, 1, -1)], torch.randn(5, 3), edge_index_dict, 0, symmetric_edges=False)
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
