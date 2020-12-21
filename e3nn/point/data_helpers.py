# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import collections
from functools import partial

import torch
import torch_geometric as tg
import numpy as np
import ase.neighborlist
from pymatgen.core.structure import Structure

from e3nn import o3, rs
from e3nn.tensor import SphericalTensor
from e3nn.util.deprecation import deprecated


class DataNeighbors(tg.data.Data):
    def __init__(self, x, pos, r_max, cell = None, pbc = False, self_interaction=True, **kwargs):
        edge_index, edge_attr = _neighbor_list_and_relative_vec(
            pos,
            r_max,
            self_interaction = self_interaction,
            cell = cell,
            pbc = pbc
        )
        if cell is not None:
            # For compatability: the old DataPeriodicNeighbors put the cell
            # in the Data object as `lattice`.
            kwargs['lattice'] = cell
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)

    @classmethod
    def from_ase(cls, atoms, r_max, features = None, **kwargs):
        if features is None:
            _, species_ids = np.unique(atoms.get_atomic_numbers(), return_inverse = True)
            features = torch.nn.functional.one_hot(torch.as_tensor(species_ids)).to(dtype = torch.get_default_dtype())
        return cls(
            x = features,
            pos = atoms.positions,
            r_max = r_max,
            cell = atoms.get_cell(complete = True),
            pbc = atoms.pbc,
            **kwargs
        )


class DataPeriodicNeighbors(DataNeighbors):
    def __init__(self, x, pos, lattice, r_max, self_interaction=True, **kwargs):
        super().__init__(
            x=x, pos=pos, cell=lattice, pbc = True, r_max=r_max,
            self_interaction = self_interaction, **kwargs
        )


class DataEdgeNeighbors(tg.data.Data):
    """Constructs graph to perform edge convolutions.

    Symmetric edges have not yet been implemented for this class.

    Args:
        x (torch.tensor shape [N, rs.dim(Rs_in)]): Node features.
        Rs_in (rs.TY_RS_STRICT): Representation list of input.
        pos (torch.tensor shape [N, 3]): Cartesian coordinates of nodes.
        r_max (float): Radial cutoff for edges.
        lmax (int > 0): Maximum L to use for SphericalTensor projection of radial distance vectors
        self_interaction (bool, optional): Include self interactions of nodes. Defaults to True.
        symmetric_edges (bool, optional): Constrain edge features to be symmetric in node index. Defaults to False.
        self_edge (float, optional): L=0 feature for self edges. Defaults to 1.
    """
    def __init__(self, x, Rs_in, pos, r_max, lmax,
                 self_interaction=True, symmetric_edges=False, self_edge=1., **kwargs):
        edge_index, edge_attr = _neighbor_list_and_relative_vec(pos, r_max, self_interaction)
        edge_index_dict, edge_edges, edge_edge_index = _get_edge_edges_and_index(edge_index, symmetric_edges=symmetric_edges)
        edge_edge_attr = []
        for _, edge2 in edge_edges:
            target2, source2 = edge2
            edge_edge_attr.append(
                pos[source2] - pos[target2]
            )

        edge_edge_index = torch.LongTensor(edge_edge_index).transpose(0, 1)
        edge_edge_attr = torch.stack(edge_edge_attr, dim=0)

        edge_x, Rs_in_edge = _initialize_edges(x, Rs_in, pos, edge_index_dict, lmax, self_edge=self_edge)

        super(DataEdgeNeighbors, self).__init__(
            x=x, edge_x=edge_x, edge_index=edge_index, edge_edge_index=edge_edge_index,
            edge_attr=edge_attr, edge_edge_attr=edge_edge_attr, pos=pos, Rs_in=Rs_in,
            Rs_in_edge=Rs_in_edge, edge_index_dict=edge_index_dict, **kwargs)


class DataEdgePeriodicNeighbors(tg.data.Data):
    """Constructs periodic graph to perform edge convolutions.

    symmetric_edges has not yet been implemented for this class.

    Args:
        x (torch.tensor shape [N, rs.dim(Rs_in)]): Node features.
        Rs_in (rs.TY_RS_STRICT): Representation list of input.
        pos (torch.tensor shape [N, 3]): Cartesian coordinates of nodes.
        lattice (torch.tensor shape [3, 3]): Lattice vectors of unit cell.
        r_max (float): Radial cutoff for edges.
        lmax (int > 0): Maximum L to use for SphericalTensor projection of radial distance vectors
        self_interaction (bool, optional): Include self interactions of nodes. Defaults to True.
        self_edge (float, optional): L=0 feature for self edges. Defaults to 1.
    """
    def __init__(self, x, Rs_in, pos, lattice, r_max, lmax,
                 self_interaction=True, self_edge=1., **kwargs):
        edge_index, edge_attr = _neighbor_list_and_relative_vec_lattice(pos, lattice, r_max, self_interaction)
        edge_index_dict, edge_edges, edge_edge_index = _get_edge_edges_and_index(edge_index, symmetric_edges=False)
        edge_edge_attr = []
        for _, edge2 in edge_edges:
            target2, source2 = edge2
            edge_edge_attr.append(
                pos[source2] - pos[target2]
            )

        edge_edge_index = torch.LongTensor(edge_edge_index).transpose(0, 1)
        edge_edge_attr = torch.stack(edge_edge_attr, dim=0)

        edge_x, Rs_in_edge = _initialize_edges(x, Rs_in, pos, edge_index_dict, lmax, self_edge=self_edge)

        super(DataEdgePeriodicNeighbors, self).__init__(
            x=x, edge_x=edge_x, edge_index=edge_index, edge_edge_index=edge_edge_index,
            edge_attr=edge_attr, edge_edge_attr=edge_edge_attr, pos=pos, Rs_in=Rs_in,
            Rs_in_edge=Rs_in_edge, edge_index_dict=edge_index_dict, **kwargs)


def _neighbor_list_and_relative_vec(pos, r_max, self_interaction=True, cell = None, pbc = False):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be detached & on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include self-edges in the neighbor list.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_attr (torch.tensor shape [num_edges, 3]): Relative vectors corresponding to each edge.

    """
    if isinstance(pbc, bool):
        pbc = (pbc,)*3
    if cell is None:
        # ASE will "complete" this correctly.
        cell = np.zeros((3, 3))
    cell = ase.geometry.complete_cell(cell)

    first_idex, second_idex, displacements = ase.neighborlist.primitive_neighbor_list(
        'ijD',
        pbc,
        np.asarray(cell),
        np.asarray(pos),
        cutoff = r_max,
        self_interaction = self_interaction,
        use_scaled_positions = False
    )
    edge_index = torch.vstack((
        torch.LongTensor(first_idex),
        torch.LongTensor(second_idex)
    ))
    edge_attr = torch.as_tensor(displacements)
    return edge_index, edge_attr


def _neighbor_list_and_relative_vec_lattice(pos, lattice, r_max, self_interaction=True):
    """Create neighbor list and neighbor vectors based on radial cutoff and periodic lattice.

    Compatability wrapper around ``_neighbor_list_and_relative_vec``.
    Prefer to use ``_neighbor_list_and_relative_vec`` directly in new code.
    """
    return _neighbor_list_and_relative_vec(
        pos,
        r_max,
        cell = lattice,
        pbc = (True,)*3,
        self_interaction = self_interaction
    )


def _initialize_edges(x, Rs_in, pos, edge_index_dict, lmax, self_edge=1., symmetric_edges=False):
    """Initialize edge features of DataEdgeNeighbors using node features and SphericalTensor.

    Args:
        x (torch.tensor shape [N, rs.dim(Rs_in)]): Node features.
        Rs_in (rs.TY_RS_STRICT): Representation list of input.
        pos (torch.tensor shape [N, 3]): Cartesian coordinates of nodes.
        edge_index (torch.LongTensor shape [2, num_edges]): Edges described by index of node target then node source.
        lmax (int > 0): Maximum L to use for SphericalTensor projection of radial distance vectors
        self_edge (float, optional): L=0 feature for self edges. Defaults to 1.
        symmetric_edges (bool, optional): Constrain edge features to be symmetric in node index. Defaults to False

    Returns:
        edge_x: Edge features.
        Rs_edge (rs.TY_RS_STRICT): Representation list of edge features.
    """
    edge_x = []
    if symmetric_edges:
        Rs, Q = rs.reduce_tensor('ij=ji', i=Rs_in)
    else:
        Rs, Q = rs.reduce_tensor('ij', i=Rs_in, j=Rs_in)
    Q = Q.reshape(-1, rs.dim(Rs_in), rs.dim(Rs_in))
    Rs_sph = [(1, l, (-1)**l) for l in range(lmax + 1)]
    tp_kernel = rs.TensorProduct(Rs, Rs_sph, o3.selection_rule)
    keys, values = list(zip(*edge_index_dict.items()))
    sorted_edges = sorted(zip(keys, values), key=lambda x: x[1])
    for (target, source), _ in sorted_edges:
        Ia = x[target]
        Ib = x[source]
        vector = (pos[source] - pos[target]).reshape(-1, 3)
        if torch.allclose(vector, torch.zeros(vector.shape)):
            signal = torch.zeros(rs.dim(Rs_sph))
            signal[0] = self_edge
        else:
            signal = SphericalTensor.from_geometry(vector, lmax=lmax).signal
            if symmetric_edges:
                signal += SphericalTensor.from_geometry(-vector, lmax=lmax).signal
                signal *= 0.5
        output = torch.einsum('kij,i,j->k', Q, Ia, Ib)
        output = tp_kernel(output, signal)
        edge_x.append(output)
    edge_x = torch.stack(edge_x, dim=0)
    return edge_x, tp_kernel.Rs_out


def _get_edge_edges_and_index(edge_index, symmetric_edges=False):
    """Given edge_index, construct edge_edges and edge_edge_index.

    Args:
        edge_index (torch.LongTensor shape [2, num_edges]): Edges described by index of node target then node source.
        symmetric_edges (bool, optional): Constrain edge features to be symmetric in node index. Defaults to False

    Returns:
        edge_index_dict: Dictionary of edge in terms of node indices and edge index.
        edge_edges: Pairs of edges over which to do edge convolutions using node indices. [num_edge_edges, 2, 2]
        edge_edge_index: Pairs of edges over which to do edge convolutions using edge indices. [num_edge_edges, 2]
    """
    edge_edges = []
    for target1, source1 in edge_index.transpose(1, 0).numpy():
        for target2, source2 in edge_index.transpose(1, 0).numpy():
            if target1 == target2:
                edge_edges.append(
                    [[target1, source1], [target2, source2]]
                )
    if symmetric_edges:
        distinct_edges = sorted(set(map(tuple,
                                        torch.sort(edge_index.transpose(1, 0),
                                                   dim=-1)[0].numpy().tolist())))
        print(distinct_edges)
        edge_index_dict = collections.OrderedDict(zip(distinct_edges, range(len(distinct_edges))))
        edge_edge_index = [
            [edge_index_dict[tuple(sorted(edge1))], edge_index_dict[tuple(sorted(edge2))]]
            for edge1, edge2 in edge_edges
        ]
    else:
        edge_index_dict = collections.OrderedDict(zip(map(tuple, edge_index.transpose(1, 0).numpy()), range(edge_index.shape[-1])))
        edge_edge_index = [
            [edge_index_dict[tuple(edge1)], edge_index_dict[tuple(edge2)]]
            for edge1, edge2 in edge_edges
        ]
    return edge_index_dict, edge_edges, edge_edge_index
