# pylint: disable=no-member, not-callable, missing-docstring, line-too-long, invalid-name
from os import mkdir
from os.path import isdir, isfile, join
from shutil import rmtree

import numpy as np
import pymatgen
import torch
from torch.utils.data import Dataset
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
    def __init__(self, root, max_radius, material_properties=None, bs_pad=64):
        """
        :param root: string, path to the root directory of dataset
        :param max_radius: float, radius of sphere
        :param material_properties: (optional) list of file paths containing additional properties, one type of property per file
        :param bs_pad: integer, length of fixed size vector in which we embed bs (single list)
        """
        preprocessed_dir = join(root, 'preprocessed')
        preprocessed_radius_dir = join(preprocessed_dir, f'max_radius_{max_radius}')

        if isdir(preprocessed_radius_dir) and not all(isfile(join(preprocessed_radius_dir, f)) for f in ['radii.pth', 'bs.pth', 'partitions.pth']):
            rmtree(preprocessed_radius_dir)
            CrystalCIF.preprocess(root, max_radius, bs_pad)
        elif not isdir(preprocessed_radius_dir):
            CrystalCIF.preprocess(root, max_radius, bs_pad)
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
        properties = None if self.properties is None else self.properties[item_id]
        radii_start, radii_end, bs_start, bs_end = self.partitions[item_id]
        return self.names[item_id], self.radii[radii_start:radii_end], self.bs[bs_start:bs_end], self.geometries[item_id], \
            self.atomic_charges[item_id], self.lattice_params[item_id], properties

    def __len__(self):
        return self.size

    @staticmethod
    def preprocess(root, max_radius=None, bs_pad=64):
        """
        Allows calls without class instance: CrystalCIF.preprocess(...).
        :param root: string, path to the root directory of dataset
        :param max_radius: float, (optional) radius of sphere
        :param bs_pad: integer, (unused for max_radius=None) length of fixed size vector in which we embed bs (single list)

        Notes:
            Let there be bs_lists [17, 18, 19, 24, 25] and [1, 2, 5, 6], then with bs_pad = 8 they will be stacked in tensor as follows:
            [[5, 17, 18, 19, 24, 25, 0, 0],
             [4,  1,  2,  5,  6,  0, 0, 0]]
            where first element in each row is a number n of relevant elements in it, next n elements are those copied from the lists,
            and rest padded with zeros for length to match bs_pad.

            Pros (over list of lists): has smaller overhead from structure meta-data, occupies contingent memory area.
            Cons (over list of lists): additionally stores lengths and pad elements.
        """
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

        # TODO: preallocate memory for tensors and write into it, instead of using lists.
        #  Can save ~30% on peak memory, but requires proper reallocation on the go for radii as shape isn't known beforehand.

        site_a_coords_list = []
        atomic_charges_list = []
        lattice_params_list = []

        if max_radius:
            bs_list = []
            radii_list = []
            partitions_list = []
        # endregion

        # region 2. Process
        radii_partition_start = 0
        bs_partition_start = 0
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
                for site_a_coords in site_a_coords_entry:
                    nei = structure.get_sites_in_sphere(site_a_coords.numpy(), max_radius, include_index=True)
                    assert nei, f"Encountered empty nei for {file_rel_path}: {site_a_coords}"

                    # save storage/RAM, transfer of small parts to the long "register" (for indexing) on the fly is comp. cheap
                    bs_entry = np.zeros(bs_pad, dtype=np.short)
                    bs_entry_data = [entry[2] for entry in nei]
                    bs_entry_data_len = len(bs_entry_data)
                    assert bs_entry_data_len < bs_pad, f"Encountered bs vector ({bs_entry_data_len}) longer than provided bs_pad ({bs_pad})"

                    bs_entry[0] = bs_entry_data_len                                                 # store number of meaningful entries as a first element
                    bs_entry[1:1 + bs_entry_data_len] = bs_entry_data                                 # store said meaningful entries in consecutive cells
                    bs_proxy_list.append(bs_entry)

                    site_b_coords = np.array([entry[0].coords for entry in nei])                    # [r_part_a, 3]
                    site_a_coords = np.array(site_a_coords).reshape(1, 3)                           # [1, 3]
                    radii_proxy_list.append(site_b_coords - site_a_coords)                          # implicit broadcasting of site_a_coords

                radii_proxy = np.concatenate(radii_proxy_list)                                      # [r, 3]
                radii_proxy[np.linalg.norm(radii_proxy, ord=2, axis=-1) < 1e-10] = 0.
                radii_list.append(radii_proxy)

                bs_list.append(bs_proxy_list)

                partitions_list.append((radii_partition_start, radii_partition_start + radii_proxy.shape[0],
                                        bs_partition_start, bs_partition_start + len(bs_proxy_list)))
                radii_partition_start += radii_proxy.shape[0]
                bs_partition_start += len(bs_proxy_list)
        # endregion

        # region 3. Post-process
        lattice_params = torch.tensor(lattice_params_list)
        del lattice_params_list

        if max_radius:
            bs = torch.from_numpy(np.concatenate(bs_list))
            del bs_list

            radii = torch.from_numpy(np.concatenate(radii_list))
            del radii_list

            partitions = torch.tensor(partitions_list, dtype=torch.long)
            del partitions_list
        # endregion

        # region 4. Store
        torch.save(site_a_coords_list, join(preprocessed_dir, 'geometries.pth'))                    # list of z tensors [a_i, 3]            - xyz
        torch.save(atomic_charges_list, join(preprocessed_dir, 'atomic_charges.pth'))               # list of z tensors [a_i]
        torch.save(lattice_params, join(preprocessed_dir, 'lattice_params.pth'))                    # [z, 6]                                - xyz and angles (in degrees)
        del site_a_coords_list, atomic_charges_list, lattice_params

        if max_radius:
            torch.save(bs, join(max_radius_dir, 'bs.pth'))                                          # tensor [sum(a_i), bs_pad]
            del bs

            torch.save(radii, join(max_radius_dir, 'radii.pth'))                                    # tensor [sum(r_i), 3]                  - xyz
            del radii

            # tensor [z, 4]                         - start/end of radii slice and bs slice corresponding to z
            torch.save(partitions, join(max_radius_dir, 'partitions.pth'))
            del partitions
        # endregion
