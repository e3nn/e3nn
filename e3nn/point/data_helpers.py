import torch
import torch_geometric as tg


def neighbor_list_and_relative_vec(pos, r_max, self_interaction=True):
    """
    Create neighbor list (edge_index) and relative vectors (edge_attr)
    based on radial cutoff.

    :param pos: torch.tensor of coordinates with shape (N, 3)
    :param r_max: float of radial cutoff
    :param self_interaction: whether or not to include self edge
    """
    from ase import Atoms, neighborlist
    N, _ = pos.shape
    atoms = Atoms(symbols=['H'] * N, positions=pos)
    nl = neighborlist.NeighborList(
        [r_max] * N,
        self_interaction=self_interaction,
        bothways=True,
    )
    nl.update(atoms)

    nei_list = []
    geo_list = []

    for i, p in enumerate(pos):
        indices = nl.get_neighbors(i)[0]
        if self_interaction:
            indices = indices[:-1]  # Remove extra self edge
        cart = pos[indices]
        indices = torch.LongTensor([[i, target] for target in indices])
        dist = cart - p
        nei_list.append(indices)
        geo_list.append(dist)
    return torch.cat(nei_list, dim=0).transpose(1, 0), torch.cat(geo_list, dim=0)


def neighbor_list_and_relative_vec_lattice(pos, lattice, r_max, self_interaction=True):
    """
    Create neighbor list (edge_index) and relative vectors (edge_attr)
    based on radial cutoff and periodic lattice.

    :param pos: torch.tensor of coordinates with shape (N, 3)
    :param r_max: float of radial cutoff
    :param self_interaction: whether or not to include self edge

    """
    from pymatgen.core.structure import Structure
    N, _ = pos.shape
    structure = Structure(lattice, ['H'] * N, pos, coords_are_cartesian=True)

    nei_list = []
    geo_list = []

    if self_interaction:
        r_min = 0.
    else:
        r_min = 1e-8

    neighbors = structure.get_all_neighbors(
        r_max,
        include_index=True,
        include_image=True,
        numerical_tol=r_min
    )
    for i, (site, neis) in enumerate(zip(structure, neighbors)):
        indices, cart = zip(*[(n.index, n.coords) for n in neis])
        cart = torch.tensor(cart)
        indices = torch.LongTensor([[i, target] for target in indices])
        dist = cart - torch.tensor(site.coords)
        nei_list.append(indices)
        geo_list.append(dist)
    return torch.cat(nei_list, dim=0).transpose(1, 0), torch.cat(geo_list, dim=0)


class DataNeighbors(tg.data.Data):
    def __init__(self, x, Rs_in, pos, r_max, self_interaction=True, **kwargs):
        edge_index, edge_attr = neighbor_list_and_relative_vec(
            pos, r_max, self_interaction)
        super(DataNeighbors, self).__init__(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, Rs_in=Rs_in, **kwargs)


class DataPeriodicNeighbors(tg.data.Data):
    def __init__(self, x, Rs_in, pos, lattice, r_max, self_interaction=True, **kwargs):
        edge_index, edge_attr = neighbor_list_and_relative_vec_lattice(
            pos, lattice, r_max, self_interaction)
        super(DataPeriodicNeighbors, self).__init__(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, lattice=lattice, Rs_in=Rs_in, **kwargs)
