# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch

from e3nn import SO3


class Convolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, out_geometry=None, n_norm=1):
        """
        :param features:     tensor [batch,  in_point, channel]
        :param geometry:     tensor [batch,  in_point, xyz]
        :param out_geometry: tensor [batch, out_point, xyz]
        :return:             tensor [batch, out_point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        if out_geometry is None:
            out_geometry = geometry
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = out_geometry.unsqueeze(2)  # [batch, a, 1, xyz]
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
    def __init__(self, Kernel, Rs_in, Rs_out, max_radius):
        super().__init__()
        self.max_radius = max_radius
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, lattice, n_norm=None):
        """
        :param features:   tensor [batch, point, channel]
        :param geometry:   tensor [batch, point, xyz]
        :param lattice:    pymatgen.Lattice
        :param n_norm:     float
        :return:           tensor [batch, point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features ({}) and geometry ({}) sizes of the first two dimensions should match".format(features.size(), geometry.size())
        import pymatgen
        import numpy as np

        batch_size, points_num = features.size(0), features.size(1)
        in_channels, out_channels = SO3.dimRs(self.kernel.Rs_in), SO3.dimRs(self.kernel.Rs_out)

        geometry = geometry.cpu().numpy()
        features = features.view(batch_size, points_num * in_channels)                                               # [z, b*j]
        out = features.new_zeros(batch_size, points_num, out_channels)                                               # [z, a, i]

        for z, geo in enumerate(geometry):
            structure = pymatgen.Structure(lattice, ["H"] * points_num, geo, coords_are_cartesian=True)
            bs_list = []
            radius_list = []

            for a, site_a in enumerate(structure):
                nei = structure.get_sites_in_sphere(site_a.coords, self.max_radius, include_index=True, include_image=True)
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


class PeriodicConvolutionPrep(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, Kernel):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, radii, bs_slice):
        """
        :param features: [point, channel_in]
        :param radii: [r, xyz]
        :param bs_slice: [point, bs_pad]
        :return: out: [point, channel_out]
        """
        kernels = self.kernel(radii)                                                                # [r, i, j]
        out = PeriodicConvolutionFunc.apply(kernels, bs_slice, features)
        return out


class PeriodicConvolutionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernels, bs_slice, features):
        ctx.save_for_backward(kernels, bs_slice, features)                                          # save input for backward pass

        points_num = features.size(0)
        in_channels, out_channels = kernels.size(2), kernels.size(1)

        features = features.view(points_num * in_channels)                                          # [b*j]
        bs_slice = bs_slice.long()                                                                  # comes as short int, but only long int can be used as index

        out = features.new_zeros(points_num, out_channels)                                          # [a, i]
        ks_start = 0                                                                                # kernels stacked flat - indicate where block of interest begins
        for a, bs in enumerate(bs_slice):
            bs = bs[1:1+bs[0]]                                                                      # select data from padded vector
            k_b = features.new_zeros(points_num, out_channels, in_channels)                         # [b, i, j]
            k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start + bs.size(0)])
            ks_start += bs.size(0)
            k_b = k_b.transpose(0, 1).contiguous().view(out_channels, points_num * in_channels)     # [i, b*j]

            out[a] = torch.mv(k_b, features)                                                        # [i, b*j] @ [b*j] -> [i]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernels, bs_slice, features = ctx.saved_tensors                                             # restore input from context
        del ctx                                                                                     # pros: allows to decrease peak memory usage (del kernels); cons: backward cannot be called more than once!

        points_num, out_channels = grad_output.size(0), grad_output.size(1)
        in_channels = features.size(1)

        bs_slice = bs_slice.long()                                                                  # comes as short int, but only long int can be used as index

        # region features_grad[b, j] = sum_(a, i){ k_b[a, b, i, j] * grad_output[a, i] }
        features_grad = torch.zeros_like(features).view(points_num * in_channels)                   # [b*j]
        ks_start = 0
        for a, bs in enumerate(bs_slice):
            bs = bs[1:1+bs[0]]                                                                      # select data from padded vector
            k_b = features.new_zeros(points_num, out_channels, in_channels)
            k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start + bs.size(0)])
            ks_start += bs.size(0)
            k_b = k_b.transpose(0, 1).contiguous().view(out_channels, points_num * in_channels)     # [i, b*j]

            features_grad += (k_b * grad_output[a].unsqueeze(1).expand_as(k_b)).sum(dim=0)          # sum(i){ [i, b*j] * ([i] -> [i, b*j]) } -> [b*j]

        features_grad = features_grad.view(points_num, in_channels)                                 # [b, j]
        # endregion

        del kernels                                                                                 # no longer needed, frees the same amount of memory as kernels_grad is going to take

        # region kernels_grad[r, i, j] = features[fb(r), j] * grad_output[fa(r), i].  Each r correspond to unique pair (a, b)
        fa, fb = torch.tensor([(a, b.item()) for a, bs in enumerate(bs_slice) for b in bs[1:1+bs[0]]], dtype=torch.long, device=features.device).t()
        kernels_grad = features[fb].unsqueeze(1) * grad_output[fa].unsqueeze(2)                     # implicit broadcasting: [r, 1, j] and [r, i, 1] -> [r, i ,j]
        # endregion

        return kernels_grad, None, features_grad                                                    # No gradients for bs_slice
