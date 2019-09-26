# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long
import torch


class Convolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, n_norm=1):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra)  # [batch, a, b, i, j]
        k.div_(n_norm ** 0.5)
        return torch.einsum("zabij,zbj->zai", (k, features))  # [batch, point, channel]


class PairConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out * 6)

    def forward(self, features, geometry, n_norm=1):
        """
        :param features: tensor [batch, c, d, channel]
        :param geometry: tensor [batch, a, xyz]
        :return:         tensor [batch, a, b, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra)  # [batch, a, b, 6 * i, j]
        k.div_((6 * n_norm ** 2) ** 0.5)
        k1, k2, k3, k4, k5, k6 = k.split(k.size(3) // 6, 3)
        out = 0
        out += torch.einsum("zabij,zcdj->zabi", (k1, features))  # [batch, a, b, channel]
        out += torch.einsum("zacij,zcdj->zai", (k2, features)).unsqueeze(2)  # [batch, a, b, channel]
        out += torch.einsum("zadij,zcdj->zai", (k3, features)).unsqueeze(2)  # [batch, a, b, channel]
        out += torch.einsum("zbcij,zcdj->zbi", (k4, features)).unsqueeze(1)  # [batch, a, b, channel]
        out += torch.einsum("zbdij,zcdj->zbi", (k5, features)).unsqueeze(1)  # [batch, a, b, channel]
        out += torch.einsum("zcdij,zcdj->zi", (k6, features)).unsqueeze(1).unsqueeze(2)  # [batch, a, b, channel]
        return out


class PairConvolution2(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel1 = Kernel(Rs_in, Rs_out)
        self.kernel2 = Kernel(Rs_out, Rs_out)

    def forward(self, features, geometry, n_norm=1):
        """
        :param features: tensor [batch, c, d, channel]
        :param geometry: tensor [batch, a, xyz]
        :return:         tensor [batch, a, b, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k1 = self.kernel1(rb - ra)  # [batch, a, b, i, j]
        k2 = self.kernel2(rb - ra)  # [batch, a, b, i, j]
        k1.div_(n_norm)
        return torch.einsum("zacij,zbdjk,zcdk->zabi", (k2, k1, features))  # [batch, a, b, channel]


class ApplyKernel(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, _n_norm=1):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra)  # [batch, a, b, i, j]
        return torch.einsum("zabij,zbj->zabi", (k, features))  # [batch, point, point, channel]


# TODO optimize this class
class NeighborsConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out, radius):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)
        self.radius = radius

    def forward(self, features, geometry, n_norm=1):
        """
        :param features: tensor [batch, point, channel]
        :param geometry: tensor [batch, point, xyz]
        :return:         tensor [batch, point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        batch, n, _ = geometry.size()

        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        diff = rb - ra  # [batch, a, b, xyz]
        norm = diff.norm(2, dim=-1).view(batch * n, n)  # [batch * a, b]

        neighbors = [
            (norm[i] < self.radius).nonzero().flatten()
            for i in range(batch * n)
        ]

        k = max(len(nei) for nei in neighbors)
        rel_mask = features.new_zeros(batch * n, k)
        for i, nei in enumerate(neighbors):
            rel_mask[i, :len(nei)] = 1
        rel_mask = rel_mask.view(batch, n, k)  # [batch, a, b]

        neighbors = torch.stack([
            torch.cat([nei, nei.new_zeros(k - len(nei))])
            for nei in neighbors
        ])
        neighbors = neighbors.view(batch, n, k)  # [batch, a, b]

        rb = geometry[torch.arange(batch).view(-1, 1, 1), neighbors, :]  # [batch, a, b, xyz]
        ra = geometry[torch.arange(batch).view(-1, 1, 1), torch.arange(n).view(1, -1, 1), :]  # [batch, a, 1, xyz]
        diff = rb - ra  # [batch, a, b, xyz]

        neighbor_features = features[torch.arange(batch).view(-1, 1, 1), neighbors, :]  # [batch, a, b, j]

        k = self.kernel(diff)  # [batch, a, b, i, j]
        k.div_(n_norm ** 0.5)
        output = torch.einsum('zab,zabij,zabj->zai', (rel_mask, k, neighbor_features))  # [batch, a, i]

        return output


class PeriodicConvolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, lattice, max_radius, n_norm=None):
        """
        :param features:   tensor [batch, point, channel]
        :param geometry:   tensor [batch, point, xyz]
        :param lattice:    pymatgen.Lattice
        :param max_radius: float
        :param n_norm:     float
        :return:           tensor [batch, point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features ({}) and geometry ({}) sizes of the first two dimensions should match".format(features.size(), geometry.size())
        import pymatgen
        import numpy as np

        batch_size, points_num = features.size(0), features.size(1)
        in_channels, out_channels = self.kernel.n_in, self.kernel.n_out

        geometry = geometry.cpu().numpy()
        features = features.view(batch_size, points_num * in_channels)                                               # [z, b*j]
        out = features.new_zeros(batch_size, points_num, out_channels)                                               # [z, a, i]

        for z, geo in enumerate(geometry):
            structure = pymatgen.Structure(lattice, ["H"] * points_num, geo, coords_are_cartesian=True)
            bs_list = []
            radius_list = []

            for a, site_a in enumerate(structure):
                nei = structure.get_sites_in_sphere(site_a.coords, max_radius, include_index=True, include_image=True)
                if nei:
                    bs_entry = torch.tensor([entry[2] for entry in nei], dtype=torch.long, device=features.device)   # [r_part_a]
                    bs_list.append(bs_entry)

                    site_b_coords = np.array([entry[0].coords for entry in nei])                                     # [r_part_a, 3]
                    site_a_coords = np.array(site_a.coords).reshape(1, 3)                                            # [1, 3]
                    radius_list.append(site_b_coords - site_a_coords)                                                # implicit broadcasting of site_a_coords

            radius = np.concatenate(radius_list)                                                                     # [r, 3]
            radius[np.linalg.norm(radius, ord=2, axis=-1) < 1e-10] = 0.

            radius = torch.from_numpy(radius).to(features.device)                                                    # [r, 3]
            kernels = self.kernel(radius)                                                                            # [r, i, j]

            ks_start = 0                                                                                             # kernels stacked flat - indicate where block of interest begins
            for a, bs in enumerate(bs_list):
                k_b = features.new_zeros(points_num, out_channels, in_channels)                                      # [b, i, j]
                k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start+bs.size(0)])
                ks_start += bs.size(0)
                k_b = k_b.transpose(0, 1).contiguous().view(out_channels, points_num * in_channels)                  # [i, b*j]

                out[z, a] = torch.mv(k_b, features[z])                                                               # [i, b*j] @ [b*j] -> [i]

        if n_norm:
            out.div_(n_norm ** 0.5)

        return out                                                                                                   # [batch, point, channel]
