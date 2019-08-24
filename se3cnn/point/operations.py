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
        out = torch.einsum("zabij,zcdj->zabi", (k1, features))  # [batch, a, b, channel]
        out += torch.einsum("zacij,zcdj->zai", (k2, features)).unsqueeze(2)  # [batch, a, b, channel]
        out += torch.einsum("zadij,zcdj->zai", (k3, features)).unsqueeze(2)  # [batch, a, b, channel]
        out += torch.einsum("zbcij,zcdj->zbi", (k4, features)).unsqueeze(1)  # [batch, a, b, channel]
        out += torch.einsum("zbdij,zcdj->zbi", (k5, features)).unsqueeze(1)  # [batch, a, b, channel]
        out += torch.einsum("zcdij,zcdj->zi", (k6, features)).unsqueeze(1).unsqueeze(2)  # [batch, a, b, channel]
        return out


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

    def forward(self, features, geometry, lattice, max_radius, n_norm=1):
        """
        :param features:   tensor [batch, point, channel]
        :param geometry:   tensor [batch, point, xyz]
        :param lattice:    pymatgen.Lattice
        :param max_radius: float
        :return:           tensor [batch, point, channel]
        """
        import pymatgen
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())

        radius = []
        bs = []
        ns = []
        for geo in geometry:
            structure = pymatgen.Structure(lattice, ["H"] * len(geo), geo.cpu().numpy(), coords_are_cartesian=True)

            for site_a in structure:
                nei = structure.get_sites_in_sphere(site_a.coords, max_radius, include_index=True, include_image=True)
                ns.append(len(nei))
                for site_b, _, b, _ in nei:
                    radius.append(geometry.new_tensor(site_b.coords - site_a.coords))
                    bs.append(b)

        radius = torch.stack(radius)
        radius[radius.norm(2, -1) < 1e-10] = 0
        kernels = self.kernel(radius)  # [r, i, j]

        ns = iter(ns)
        bs = iter(bs)
        ks = iter(kernels)

        k_z = []
        for _ in range(geometry.size(0)):
            k_a = []
            for _ in range(geometry.size(1)):
                k_b = [torch.zeros_like(kernels[0]) for _ in range(geometry.size(1))]
                for _ in range(next(ns)):
                    k_b[next(bs)] += next(ks)  # [i, j]
                k_b = torch.stack(k_b)  # [b, i, j]
                k_a.append(k_b)
            k_z.append(torch.stack(k_a))  # [a, b, i, j]
        k = torch.stack(k_z)  # [z, a, b, i, j]
        k.div_(n_norm ** 0.5)

        return torch.einsum("zabij,zbj->zai", (k, features))  # [point, channel]
