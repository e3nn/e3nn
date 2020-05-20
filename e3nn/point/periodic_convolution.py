# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import numpy as np
import torch

import pymatgen
from e3nn import rs


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

        batch_size, points_num = features.size(0), features.size(1)
        in_channels, out_channels = rs.dim(self.kernel.Rs_in), rs.dim(self.kernel.Rs_out)

        geometry = geometry.cpu().numpy()
        features = features.reshape(batch_size, points_num * in_channels)                                               # [z, b*j]
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

            # kernels stacked flat - indicate where block of interest begins
            ks_start = 0
            for a, bs in enumerate(bs_list):
                k_b = features.new_zeros(points_num, out_channels, in_channels)                                      # [b, i, j]
                k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start + bs.size(0)])
                ks_start += bs.size(0)
                k_b = k_b.transpose(0, 1).reshape(out_channels, points_num * in_channels)                  # [i, b*j]

                out[z, a] = torch.mv(k_b, features[z])                                                               # [i, b*j] @ [b*j] -> [i]

        if n_norm:
            out.div_(n_norm ** 0.5)

        return out                                                                                                   # [batch, point, channel]


class PeriodicConvolutionPrep(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, Kernel):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, radii, bs_slice, n_norm=None):
        """
        :param features: [point, channel_in]
        :param radii: [r, xyz]
        :param bs_slice: [point, bs_pad]
        :return: out: [point, channel_out]
        """
        kernels = self.kernel(radii)                                                                # [r, i, j]
        out = PeriodicConvolutionFunc.apply(kernels, bs_slice, features)

        if n_norm:
            out.div_(n_norm ** 0.5)

        return out


class PeriodicConvolutionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernels, bs_slice, features):
        ctx.save_for_backward(kernels, bs_slice, features)                                          # save input for backward pass

        points_num = features.size(0)
        in_channels, out_channels = kernels.size(2), kernels.size(1)

        features = features.reshape(points_num * in_channels)                                          # [b*j]
        bs_slice = bs_slice.long()                                                                  # comes as short int, but only long int can be used as index

        out = features.new_zeros(points_num, out_channels)                                          # [a, i]
        ks_start = 0                                                                                # kernels stacked flat - indicate where block of interest begins
        for a, bs in enumerate(bs_slice):
            bs = bs[1:1 + bs[0]]                                                                      # select data from padded vector
            k_b = features.new_zeros(points_num, out_channels, in_channels)                         # [b, i, j]
            k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start + bs.size(0)])
            ks_start += bs.size(0)
            k_b = k_b.transpose(0, 1).reshape(out_channels, points_num * in_channels)     # [i, b*j]

            out[a] = torch.mv(k_b, features)                                                        # [i, b*j] @ [b*j] -> [i]

        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        kernels, bs_slice, features = ctx.saved_tensors                                             # restore input from context
        # pros: allows to decrease peak memory usage (del kernels); cons: backward cannot be called more than once!
        del ctx

        points_num, out_channels = grad_output.size(0), grad_output.size(1)
        in_channels = features.size(1)

        bs_slice = bs_slice.long()                                                                  # comes as short int, but only long int can be used as index

        # region features_grad[b, j] = sum_(a, i){ k_b[a, b, i, j] * grad_output[a, i] }
        features_grad = torch.zeros_like(features).reshape(points_num * in_channels)                   # [b*j]
        ks_start = 0
        for a, bs in enumerate(bs_slice):
            bs = bs[1:1 + bs[0]]                                                                      # select data from padded vector
            k_b = features.new_zeros(points_num, out_channels, in_channels)
            k_b.index_add_(dim=0, index=bs, source=kernels[ks_start:ks_start + bs.size(0)])
            ks_start += bs.size(0)
            k_b = k_b.transpose(0, 1).reshape(out_channels, points_num * in_channels)     # [i, b*j]

            features_grad += (k_b * grad_output[a].unsqueeze(1).expand_as(k_b)).sum(dim=0)          # sum(i){ [i, b*j] * ([i] -> [i, b*j]) } -> [b*j]

        features_grad = features_grad.reshape(points_num, in_channels)                                 # [b, j]
        # endregion

        # no longer needed, frees the same amount of memory as kernels_grad is going to take
        del kernels

        # region kernels_grad[r, i, j] = features[fb(r), j] * grad_output[fa(r), i].  Each r correspond to unique pair (a, b)
        fa, fb = torch.tensor([(a, b.item()) for a, bs in enumerate(bs_slice) for b in bs[1:1 + bs[0]]], dtype=torch.long, device=features.device).t()
        kernels_grad = features[fb].unsqueeze(1) * grad_output[fa].unsqueeze(2)                     # implicit broadcasting: [r, 1, j] and [r, i, 1] -> [r, i ,j]
        # endregion

        return kernels_grad, None, features_grad                                                    # No gradients for bs_slice
