from os.path import join, isfile, isdir
from os import mkdir
from shutil import rmtree

import torch
from torch.utils.data import Dataset

import numpy as np
import pymatgen

from tqdm import tqdm


class CrystalCIF(Dataset):
    """
    root
    |_______index.npy
    |_______names.npy
    |_______property0.pth
    |_______property1.pth
    |_______cif
    |        |____name0.cif
    |        |____name1.cif
    |
    |_______preprocessed (script creates this folder and its contents)
             |____geometries.pth
             |____atomic_charges.pth
             |____lattice_params.pth
             |____max_radius_{max_radius_value}
                    |_____radii.pth
                    |_____bs.pth
                    |_____partitions.pth
    """
    def __init__(self, root, max_radius, material_properties=None):
        """
        :param root: string path to the root directory of dataset
        :param max_radius: float radius of sphere (?)
        :param material_properties: (optional) list of file paths containing additional properties, one type of property per file
        """
        preprocessed_dir = join(root, 'preprocessed')
        preprocessed_radius_dir = join(preprocessed_dir, f'max_radius_{max_radius}')

        if (
                isdir(preprocessed_radius_dir)
                and (not isfile(join(preprocessed_radius_dir, 'radii.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'bs.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'partitions.pth')))
        ):
            rmtree(preprocessed_radius_dir)
            CrystalCIF.preprocess(root, max_radius)
        elif not isdir(preprocessed_radius_dir):
            CrystalCIF.preprocess(root, max_radius)
        else:
            pass

        self.names = np.load(join(root, 'names.npy'))
        self.size = len(self.names)
        self.geometries = torch.load(join(preprocessed_dir, 'geometries.pth'))
        self.atomic_charges = torch.load(join(preprocessed_dir, 'atomic_charges.pth'))
        self.lattice_params = torch.load(join(preprocessed_dir, 'lattice_params.pth'))

        self.radii = torch.load(join(preprocessed_radius_dir, 'radii.pth'))
        self.bs = torch.load(join(preprocessed_radius_dir, 'bs.pth'))
        self.partitions = torch.load(join(preprocessed_radius_dir, 'partitions.pth'))

        if material_properties:
            self.properties = torch.stack([torch.load(join(root, property_path)) for property_path in material_properties], dim=1)
        else:
            self.properties = None

    def __getitem__(self, item_id):
        start, end = self.partitions[item_id]
        properties = None if self.properties is None else self.properties[item_id]
        return self.radii[start:end], self.bs[start:end], self.geometries[item_id], self.atomic_charges[item_id], self.lattice_params[item_id], properties

    def __len__(self):
        return self.size

    @staticmethod
    def preprocess(root, max_radius=None):
        # region 0. Set up
        preprocessed_dir = join(root, 'preprocessed')
        if not isdir(preprocessed_dir):
            mkdir(preprocessed_dir)

        if max_radius:
            max_radius_dir = join(preprocessed_dir, f'max_radius_{max_radius}')
            if not isdir(max_radius_dir):
                mkdir(max_radius_dir)
        # endregion

        # region 1. Init
        index = np.load(join(root, 'index.npy'))

        site_a_coords_list = []
        atomic_charges_list = []
        lattice_params_list = []

        if max_radius:
            bs_list = []
            radii_list = []
            partitions_list = []
        # endregion

        # region 2. Process
        for file_rel_path in tqdm(index):
            structure = pymatgen.Structure.from_file((join(root, 'cif', file_rel_path)))

            site_a_coords_entry = torch.stack([torch.from_numpy(site.coords) for site in structure.sites])
            atomic_charges_entry = torch.tensor([atom.number for atom in structure.species])

            site_a_coords_list.append(site_a_coords_entry)
            atomic_charges_list.append(atomic_charges_entry)
            lattice_params_list.append(structure.lattice.abc + structure.lattice.angles)

            if max_radius:
                radii_proxy_list = []
                bs_proxy_list = []
                partition_start = 0
                for site_a_coords in site_a_coords_entry:
                    nei = structure.get_sites_in_sphere(site_a_coords.numpy(), max_radius, include_index=True)
                    if nei:
                        # TODO: storing bs as a list of tensors seems excessive, bitset would be nice, but python don't have one, maybe list of pointers and set of lists (?)
                        bs_entry = torch.tensor([entry[2] for entry in nei], dtype=torch.long)      # [r_part_a]
                        bs_proxy_list.append(bs_entry)

                        site_b_coords = np.array([entry[0].coords for entry in nei])                # [r_part_a, 3]
                        site_a_coords = np.array(site_a_coords).reshape(1, 3)                       # [1, 3]
                        radii_proxy_list.append(site_b_coords - site_a_coords)                      # implicit broadcasting of site_a_coords
                    else:
                        print(f"Encountered empty nei for {file_rel_path}: {site_a_coords}")

                radii_proxy = np.concatenate(radii_proxy_list)                                      # [r, 3]
                radii_proxy[np.linalg.norm(radii_proxy, ord=2, axis=-1) < 1e-10] = 0.
                radii_list.append(radii_proxy)
                bs_list.append(bs_proxy_list)
                partitions_list.append((partition_start, partition_start + radii_proxy.shape[0]))
                partition_start += radii_proxy.shape[0]
        # endregion

        # region 3. Post-process
        lattice_params = torch.tensor(lattice_params_list)

        if max_radius:
            radii = torch.from_numpy(np.concatenate(radii_list))
            partitions = torch.tensor(partitions_list, dtype=torch.long)
        # endregion

        # region 4. Store
        torch.save(site_a_coords_list, join(preprocessed_dir, 'geometries.pth'))                    # list of z tensors [a_i, 3]            - xyz
        torch.save(atomic_charges_list, join(preprocessed_dir, 'atomic_charges.pth'))               # list of z tensors [a_i]
        torch.save(lattice_params, join(preprocessed_dir, 'lattice_params.pth'))                    # [z, 6]                                - xyz and angles (in degrees)

        if max_radius:
            torch.save(bs_list, join(max_radius_dir, 'bs.pth'))                                     # list of z lisradii_proxy_listts of tensors [r_i]
            torch.save(radii, join(max_radius_dir, 'radii.pth'))                                    # tensor [sum(r_i), 3]                  - xyz
            torch.save(partitions, join(max_radius_dir, 'partitions.pth'))                          # tensor [z, 2]                         - start/end of radii slice corresponding to z
        # endregion
