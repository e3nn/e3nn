from os.path import join, isfile, isdir
from os import mkdir
from shutil import rmtree

import numpy as np
import torch
from torch.utils.data import Dataset


class CrystalCIF(Dataset):
    """
    root
    |_______index.npy
    |_______names.npy
    |_______property0.pth
    |_______property1.pth
    |_______cif
    |        |____0.cif
    |        |____1.cif
    |
    |_______preprocessed
             |____site_a_coord.pth
             |____atomic_charges.pth
             |____lattice_params.pth
             |____max_radius_{max_radius_value}
                    |_____radius.pth
                    |_____bs.pth
                    |_____partitioning.pth
    """
    def __init__(self, root, max_radius, material_properties=None):
        """
        :param root: string path to the root directory of dataset
        :param material_properties: (optional) list of file paths containing additional properties, one type of property per file
        """
        preprocessed_dir = join(root, 'preprocessed')                                           # check if (all) basic preprocessed data is available, if it is not - compute
        if not isdir(preprocessed_dir):
            CrystalCIF.preprocess_base(root)
        elif (
                not isfile(join(preprocessed_dir, 'site_a_coord.pth'))
                or not isfile(join(preprocessed_dir, 'atomic_charges.pth'))
                or not isfile(join(preprocessed_dir, 'lattice_params.pth'))
                or not isfile(join(preprocessed_dir, 'partitioning.pth'))
        ):
            rmtree(preprocessed_dir)                                                            # clean up corrupted structure
            CrystalCIF.preprocess_base(root)
        else:
            pass

        preprocessed_radius_dir = join(preprocessed_dir, f'max_radius_{max_radius}')            # check if data for requested max_radius is available, if it is not - compute
        if not isdir(preprocessed_radius_dir):
            CrystalCIF.preprocess_radius(preprocessed_dir, max_radius)
        elif (
                not isfile(join(preprocessed_radius_dir, 'radius.pth'))
                or not isfile(join(preprocessed_radius_dir, 'bs.pth'))
                or not isfile(join(preprocessed_radius_dir, 'partitioning.pth'))
        ):
            rmtree(preprocessed_radius_dir)                                                     # clean up corrupted structure
            CrystalCIF.preprocess_radius(root, max_radius)
        else:
            pass

        self.names = np.load(join(root, 'names.npy'))
        self.size = len(self.names)
        self.atomic_charges = torch.load(join(preprocessed_dir, 'atomic_charges.pth'))
        self.lattice_params = torch.load(join(preprocessed_dir, 'lattice_params.pth'))

        self.radius = torch.load(join(preprocessed_radius_dir, 'radius.pth'))
        self.bs = torch.load(join(preprocessed_radius_dir, 'bs.pth'))
        self.partitioning = torch.load(join(preprocessed_radius_dir, 'partitioning.pth'))

        if material_properties:
            self.properties = torch.stack([torch.load(join(root, property_path)) for property_path in material_properties], dim=1)
        else:
            self.properties = self.radius.new_zeros(self.size, 1)                                # placeholder for consistency

    def __getitem__(self, item_id):
        start, end = self.partitioning[item_id]
        return self.radius[start:end], self.bs[start:end], self.atomic_charges[item_id], self.lattice_params[item_id], self.properties[item_id]

    def __len__(self):
        return self.size

    @staticmethod
    def preprocess_base(root):
        preprocessed_dir = join(root, 'preprocessed')
        mkdir(preprocessed_dir)
        # TODO: complete
        pass

    @staticmethod
    def preprocess_radius(preprocessed_dir, max_radius):
        preprocessed_radius_dir = join(preprocessed_dir, f'max_radius_{max_radius}')
        mkdir(preprocessed_radius_dir)
        # TODO: complete
        pass



