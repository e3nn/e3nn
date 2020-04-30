# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch


class Convolution(torch.nn.Module):
    def __init__(self, Kernel, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out)

    def forward(self, features, geometry, out_geometry=None, n_norm=1,
                custom_backward_conv=False, custom_backward_kernel=False, r_eps=0):
        """
        :param features:     tensor [batch,  in_point, channel]
        :param geometry:     tensor [batch,  in_point, xyz]
        :param out_geometry: tensor [batch, out_point, xyz]
        :param n_norm: Divide kernel by sqrt(n_norm) before passing to convolution.
        :param custom_backward_conv: call ConvolutionEinsumFn rather than using automatic differentiation
        :param custom_backward_kernel: call ConvolutionEinsumFn rather than using automatic differentiation
        :return:             tensor [batch, out_point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        if out_geometry is None:
            out_geometry = geometry
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = out_geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        k = self.kernel(rb - ra, custom_backward=custom_backward_kernel, r_eps=r_eps)  # [batch, a, b, i, j]
        k.div_(n_norm ** 0.5)
        if custom_backward_conv:
            return ConvolutionEinsumFn.apply(k, features)  # [batch, point, channel]
        else:
            return torch.einsum("zabij,zbj->zai", (k, features))  # [batch, point, channel]


class ConvolutionEinsumFn(torch.autograd.Function):
    """Forward and backward written explicitly for the Convolution Function."""
    @staticmethod
    def forward(ctx, k, features):
        """
        :param k:        tensor [batch, out_point, in_point, l_out * mul_out * m_out, l_in * mul_in * m_in]
        :param features: tensor [batch, in_point, l_in * mul_in * m_in]
        """
        ctx.save_for_backward(k, features)
        return torch.einsum("zabij,zbj->zai", k, features)  # [batch, point, channel]

    @staticmethod
    def backward(ctx, grad_output):
        k, features = ctx.saved_tensors
        del ctx
        grad_k = torch.einsum("zai,zbj->zabij", grad_output, features)
        grad_features = torch.einsum("zabij,zai->zbj", k, grad_output)
        return grad_k, grad_features


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

    def forward(self, features, geometry):
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
        norm = diff.norm(2, dim=-1).reshape(batch * n, n)  # [batch * a, b]

        neighbors = [
            (norm[i] < self.radius).nonzero().flatten()
            for i in range(batch * n)
        ]

        k = max(len(nei) for nei in neighbors)
        rel_mask = features.new_zeros(batch * n, k)
        for i, nei in enumerate(neighbors):
            rel_mask[i, :len(nei)] = 1
        rel_mask = rel_mask.reshape(batch, n, k)  # [batch, a, b]

        neighbors = torch.stack([
            torch.cat([nei, nei.new_zeros(k - len(nei))])
            for nei in neighbors
        ])
        neighbors = neighbors.reshape(batch, n, k)  # [batch, a, b]

        rb = geometry[torch.arange(batch).reshape(-1, 1, 1), neighbors, :]  # [batch, a, b, xyz]
        ra = geometry[torch.arange(batch).reshape(-1, 1, 1), torch.arange(n).reshape(1, -1, 1), :]  # [batch, a, 1, xyz]
        diff = rb - ra  # [batch, a, b, xyz]

        neighbor_features = features[torch.arange(batch).reshape(-1, 1, 1), neighbors, :]  # [batch, a, b, j]

        k = self.kernel(diff)  # [batch, a, b, i, j]
        k.div_(n_norm ** 0.5)
        output = torch.einsum('zab,zabij,zabj->zai', (rel_mask, k, neighbor_features))  # [batch, a, i]

        return output
