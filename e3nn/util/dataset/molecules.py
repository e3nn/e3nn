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


def random_rotate_translate(positions, translation=1):
    while True:
        trans = torch.rand(3) * 2 - 1
        if trans.norm() <= 1:
            break
    rot = o3.rand_rot().type(torch.float32)
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
    properties_names = [
        'A GHz Rotational constant',
        'B GHz Rotational constant',
        'C GHz Rotational constant',
        'μ D Dipole moment',
        'α a^3_0 Isotropic polarizability',
        'ϵ_HOMO Ha Energy of HOMO',
        'ϵ_LUMO Ha Energy of LUMO',
        'ϵ_gap Ha Gap (ϵLUMO−ϵHOMO)',
        'R^2 a^2_0 Electronic spatial extent',
        'zpve Ha Zero point vibrational energy',
        'U_0 Ha Internal energy at 0K',
        'U Ha Internal energy at 298.15K',
        'H Ha Enthalpy at 298.15K',
        'G Ha Free energy at 298.15K',
        'C_v cal/(molK) Heat capacity at 298.15K',
    ]

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
        properties = [float(x) for x in lines[1].split()[2:]]
        # https://www.nature.com/articles/sdata201422/tables/4
        qualias = []
        positions = []
        for i in range(2, 2 + n):
            atom, x, y, z, _ = lines[i].split()
            qualias.append("CHONF".index(atom))
            positions.append([float(a.replace('*^', 'e')) for a in [x, y, z]])
        positions = [torch.tensor(pos) for pos in positions]
        if self.transform is not None:
            return self.transform(positions, qualias, properties)
        return positions, qualias, properties

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
