# pylint: disable=C,R,E1101,E1102
import torch
import torch.utils.data
import os
import scipy.io
import numpy as np
from e3nn.util.bounding_sphere import bounding_sphere
import glob
from e3nn import o3


class VoxelizeBlobs:
    def __init__(self, n_qualias, size, p):
        # for QM7 size=72 and p=0.3 works fine
        self.n_qualias = n_qualias
        self.size = size
        self.p = p
        lim = self.p * 0.5 * (self.size - 1)
        self.a = torch.linspace(-lim, lim, self.size)
        self.xx = self.a.reshape(-1, 1, 1).expand(self.size, self.size, self.size)
        self.yy = self.a.reshape(1, -1, 1).expand(self.size, self.size, self.size)
        self.zz = self.a.reshape(1, 1, -1).expand(self.size, self.size, self.size)

    def __call__(self, positions, qualias):
        fields = torch.zeros((self.n_qualias, self.size, self.size, self.size))

        for position, qualia in zip(positions, qualias):
            x, y, z = position
            x, y, z = float(x), float(y), float(z)
            density = torch.exp(-((self.xx - x)**2 + (self.yy - y)**2 + (self.zz - z)**2) / (2 * self.p**2))
            density.div_(density.sum())
            fields[qualia] += density

        return fields


def center_positions(positions):
    _radius, center = bounding_sphere([pos.numpy() for pos in positions], 1e-6)
    center = torch.tensor(center).type(torch.float32)
    return [pos - center for pos in positions]


def random_rotate_translate(positions, rotation=True, translation=1):
    while True:
        trans = torch.rand(3) * 2 - 1
        if trans.norm() <= 1:
            break
    rot = o3.rot(*torch.rand(3) * 6.2832).type(torch.float32)
    return [rot @ pos + translation * trans for pos in positions]


class QM7(torch.utils.data.Dataset):
    url = 'http://quantum-machine.org/data/qm7.mat'
    mat_file = 'qm7.mat'

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.download()
        self.transform = transform

        data = scipy.io.loadmat(os.path.join(self.root, self.mat_file))
        indices = data['P'][split]  # shape = (5, 1433)
        self.positions = data['R'][indices]  # positions, shape = (7165, 23, 3)
        self.charges = data['Z'][indices].astype(np.int32)  # charge: 5 atom types: 1, 6, 7, 8, 16, shape = (7165, 23)
        self.energies = data['T'].flatten()[indices]  # labels: atomization energies in kcal/mol, shape = (7165, )

    def __getitem__(self, index):
        positions, charges, energy = self.positions[index], self.charges[index], self.energies[index]
        positions = positions[charges > 0]
        qualias = [{1: 0, 6: 1, 7: 2, 8: 3, 16: 4}[ch] for ch in charges[charges > 0]]
        positions = [torch.tensor(pos) for pos in positions]
        if self.transform is not None:
            return self.transform(positions, qualias, energy)
        return positions, qualias, energy

    def __len__(self):
        return len(self.energies)

    def download(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        file_path = os.path.join(self.root, self.mat_file)
        if not os.path.isfile(file_path):
            print('Download...')
            from six.moves import urllib
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())


class QM9(torch.utils.data.Dataset):
    url = 'https://ndownloader.figshare.com/files/3195389'

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.files = None
        self.download()
        self.transform = transform

    def __getitem__(self, index):
        path = self.files[index]
        with open(path, "rt") as f:
            lines = f.readlines()
        n = int(lines[0])
        energy = float(lines[1].split()[12])
        qualias = []
        positions = []
        for i in range(2, 2 + n):
            atom, x, y, z, _ = lines[i].split()
            qualias.append("CHONF".index(atom))
            positions.append([float(a.replace('*^', 'e')) for a in [x, y, z]])
        positions = [torch.tensor(pos) for pos in positions]
        if self.transform is not None:
            return self.transform(positions, qualias, energy)
        return positions, qualias, energy

    def __len__(self):
        return len(self.files)

    def download(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        bz2_path = os.path.join(self.root, "data.xyz.tar.bz2")
        if not os.path.isfile(bz2_path):
            print("Download...")
            from six.moves import urllib
            data = urllib.request.urlopen(self.url)
            with open(bz2_path, 'wb') as f:
                f.write(data.read())

        tar_path = os.path.join(self.root, "data.xyz.tar")
        if not os.path.isfile(tar_path):
            print("Decompress...")
            import bz2
            file = bz2.BZ2File(bz2_path)
            with open(tar_path, 'wb') as f:
                f.write(file.read())
            file.close()

        xyz_path = os.path.join(self.root, "data.xyz")
        if not os.path.isdir(xyz_path):
            print("Extract...")
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(xyz_path)
            tar.close()

        self.files = sorted(glob.glob(os.path.join(xyz_path, "*.xyz")))
        assert len(self.files) == 133885



import torch

import os
from functools import partial
from itertools import zip_longest

from pymatgen.core.structure import Molecule
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Dataset


def to_one_hot(ind_arr, n_classes):
    one_hot = torch.zeros((len(ind_arr), n_classes))
    one_hot[torch.arange(len(ind_arr)), ind_arr] = 1.
    return one_hot


# function that constructs graph given 
def molecule_from_file_to_graph_sphere_connectivity_5_no_self_interactions(molecule_file, y):
    return molecule_from_file_to_graph_base(molecule_file, y, connectivity_func=connections_within_sphere_5_no_self_interactions)

def connections_within_sphere_5_no_self_interactions(mol):
    return connections_within_sphere(mol, r_cut=5.0, self_interaction=False)

def connections_within_sphere(mol, r_cut, self_interaction):
    edge_index_transposed = torch.tensor([[origin_site_idx, neighbor_site.index] 
                                         for origin_site_idx, site in enumerate(mol.sites) 
                                         for neighbor_site in mol.get_neighbors(site, r_cut)
                                         ], dtype=torch.int64) 
    if self_interaction:
        self_interaction_indices = torch.arange(len(mol), dtype=torch.int64).view(len(mol), 1).repeat(1,2)
        edge_index_transposed = torch.cat((edge_index_transposed, self_interaction_indices), dim=0)
    return edge_index_transposed.t()

def molecule_from_file_to_graph_base(molecule_file, y, connectivity_func):
    mol = Molecule.from_file(molecule_file)

    x = torch.tensor(mol.atomic_numbers) # further processing for features x is assumed to happen in transform
    edge_index = connectivity_func(mol)  # not necessarily sphere
    edge_attr = None
    y = y
    pos = torch.from_numpy(mol.cart_coords)

    return Data(x, edge_index, edge_attr, y, pos)


# transform function supplied to pytorch-geometric Dataset should accept as an input and produce as an output Data object (graph) only  
def molecule_graph_transform_one_hot_period_group_5_18(graph):
    return molecule_graph_transform_base(graph, featurizer=featurizer_one_hot_period_group_5_18)

def featurizer_one_hot_period_group_5_18(graph):
    return featurizer_one_hot_period_group(graph, n_periods=5, n_groups=18)

def featurizer_one_hot_period_group(graph, n_periods, n_groups):
    atoms = [Element.from_Z(atomic_number) for atomic_number in graph.x]
    periods = torch.tensor([atom.row() for atom in atoms], dtype=torch.int64) 
    groups = torch.tensor([atom.group() for atom in atoms], dtype=torch.int64)

    # shift from 1-based index to 0-based index by subtracting 1
    features = torch.cat((to_one_hot(periods-1, n_periods), to_one_hot(groups-1, n_groups)), dim=1)
    return features

def molecule_graph_transform_base(graph, featurizer):
    graph = graph.clone()
    
    # edge_attr
    origin_pos = graph.pos[graph.edge_index[0]]
    neighbor_pos = graph.pos[graph.edge_index[1]]
    graph.edge_attr = neighbor_pos - origin_pos
    
    # x
    graph.x = featurizer(graph)

    return graph


class MoleculeDataset(Dataset):
    """
    Within root folder should be a subfolder named 'raw' that contains files with molecules - one file per molecule.
    Processed files will be stored to automatically created subfolder 'processed'.  
    """
    def __init__(self, root, file_names, file_to_graph_func, properties_filepath=None, transform=None, pre_transform=None, pre_filter=None, allowed_file_types=["xyz"]+["gjf", "g03", "g09", "com", "inp"]+["json", "mson", "yaml"]+["out", "lis", "log"]+["pdb", "mol", "mdl", "sdf", "sd", "ml2", "sy2", "mol2", "cml", "mrv"]):
        self.allowed_file_types = allowed_file_types
        self.file_names = self.filter_files(file_names)
        self._processed_file_names = self.process_file_names()
        
        self.file_to_graph_func = file_to_graph_func
        self.properties_filepath = properties_filepath

        super().__init__(root, transform, pre_transform, pre_filter)

    def filter_files(self, file_names):
        # os.path.splitext(file_name)[1] - robust retrieval of the file extension
        return [file_name for file_name in file_names if os.path.splitext(file_name)[1] in self.allowed_file_types]

    def process_file_names(self):
        # os.path.splitext(file_name)[0] - robust retrieval of the file name without extension
        return [os.path.splitext(file_name)[0] + ".pth" for file_name in self.file_names]

    @property
    def raw_file_names(self):
        return self.file_names 

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        pass

    def process(self):
        properties = torch.load(self.properties_filepath) if self.properties_filepath else [None]
        for raw_path, processed_path, y in zip_longest(self.raw_paths, self.processed_paths, properties):
            graph = self.file_to_graph_func(raw_path, y)            

            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            torch.save(graph, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data
