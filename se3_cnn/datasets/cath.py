# pylint: disable=C,R,E1101
import torch
import os
import numpy as np
from scipy.stats import special_ortho_group


class Cath(torch.utils.data.Dataset):
    url = 'https://github.com/deepfold/cath_datasets/blob/master/{}?raw=true'

    def __init__(self, dataset, split, download=False, use_density=True, randomize_orientation=False):
        self.root = os.path.expanduser("cath")

        if download:
            self.download(dataset)

        self.use_density = use_density
        self.randomize_orientation = randomize_orientation

        if not self._check_exists(dataset):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data = np.load(os.path.join(self.root, dataset))
        split_start_indices = data['split_start_indices']
        split_range = list(zip(split_start_indices[0:], list(split_start_indices[1:])+[None]))[split]
        self.positions = data['positions'][split_range[0]:split_range[1]]
        self.atom_types = data['atom_types'][split_range[0]:split_range[1]]
        self.n_atoms = data['n_atoms'][split_range[0]:split_range[1]]
        self.labels = [tuple(v) if len(v) > 1 else v[0] for v in data['labels'][split_range[0]:split_range[1]]]

        self.atom_type_set = np.unique(self.atom_types[0][:self.n_atoms[0]])
        self.n_atom_types = len(self.atom_type_set)
        self.atom_type_map = dict(zip(self.atom_type_set, range(len(self.atom_type_set))))

        self.label_set = sorted(list(set(self.labels)))
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))

    def __getitem__(self, index):

        # time_stamp = timer()

        n_atoms = self.n_atoms[index]
        positions = self.positions[index][:n_atoms]
        atom_types = self.atom_types[index][:n_atoms]
        label = self.label_map[self.labels[index]]

        p = 2.0
        n = 50

        if torch.cuda.is_available():
            fields = torch.cuda.FloatTensor(*(self.n_atom_types,)+(n, n, n)).fill_(0)
        else:
            fields = torch.zeros(*(self.n_atom_types,)+(n, n, n))

        if self.randomize_orientation:
            random_rotation = special_ortho_group.rvs(3)
            positions = np.dot(random_rotation, positions.T).T

        if self.use_density:

            ## Numpy version ##
            # a = np.linspace(start=-n / 2 * p + p / 2, stop=n / 2 * p - p / 2, num=n, endpoint=True)
            # xx, yy, zz = np.meshgrid(a, a, a, indexing="ij")

            # fields_np = np.zeros((self.n_atom_types, n, n, n), dtype=np.float32)
            # for i, atom_type in enumerate(self.atom_type_set):

            #     # Extract positions with current atom type
            #     pos = positions[atom_types == atom_type]

            #     # Create grid x atom_pos grid
            #     posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
            #     posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
            #     posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))

            #     # Calculate density
            #     density = np.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (p)**2))

            #     # Normalize so each atom density sums to one
            #     density /= np.sum(density, axis=0)

            #     # Sum densities and reshape to original shape
            #     fields_np[i] = np.sum(density, axis=1).reshape(xx.shape)

            ## Pytorch version ##

            # Create linearly spaced grid
            a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)
            if torch.cuda.is_available():
                a = a.cuda()

            # Pytorch does not suppoert meshgrid - do the repeats manually
            xx = a.view(-1, 1, 1).repeat(1, len(a), len(a))
            yy = a.view(1, -1, 1).repeat(len(a), 1, len(a))
            zz = a.view(1, 1, -1).repeat(len(a), len(a), 1)

            for i, atom_type in enumerate(self.atom_type_set):

                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Transfer position vector to gpu
                pos = torch.FloatTensor(pos)
                if torch.cuda.is_available():
                    pos = pos.cuda()

                # Pytorch does not suppoert meshgrid - do the repeats manually
                # Numpy equivalent:
                # posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
                # posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
                # posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))
                xx_xx = xx.view(-1, 1).repeat(1, len(pos))
                posx_posx = pos[:, 0].contiguous().view(1, -1).repeat(len(xx.view(-1)), 1)
                yy_yy = yy.view(-1, 1).repeat(1, len(pos))
                posy_posy = pos[:, 1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
                zz_zz = zz.view(-1, 1).repeat(1, len(pos))
                posz_posz = pos[:, 2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

                # Calculate density
                sigma = 0.5*p
                density = torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

                # Normalize so each atom density sums to one
                density /= torch.sum(density, dim=0)

                # Sum densities and reshape to original shape
                fields[i] = torch.sum(density, dim=1).view(xx.shape)
        else:

            for i, atom_type in enumerate(self.atom_type_set):

                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Lookup indices and move to GPU
                indices = torch.LongTensor(np.ravel_multi_index(np.digitize(pos, a+p/2).T, dims=(n, n, n)))
                if torch.cuda.is_available():
                    indices = indices.cuda()

                # Set values
                fields[i].view(-1)[indices] = 1

        # assert((np.abs(fields.numpy() - fields_np)<0.001).all())

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), s=10*fields.numpy().reshape(-1), c=fields.numpy().reshape(-1), cmap=plt.get_cmap("Blues"))
        # plt.show()
        # plt.savefig("grid.png")

        # time_elapsed = timer() - time_stamp
        # print("Time spent on __getitem__: %.4f sec" % time_elapsed)

        return fields, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self, dataset):
        return os.path.exists(os.path.join(self.root, dataset))

    def download(self, dataset):
        from six.moves import urllib

        if self._check_exists(dataset):
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url.format(dataset))
        data = urllib.request.urlopen(self.url.format(dataset))
        file_path = os.path.join(self.root, dataset)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Done!')
