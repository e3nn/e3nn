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
    def backward(ctx, grad_output):  # pragma: no cover
        k, features = ctx.saved_tensors
        del ctx
        grad_k = torch.einsum("zai,zbj->zabij", grad_output, features)
        grad_features = torch.einsum("zabij,zai->zbj", k, grad_output)
        return grad_k, grad_features


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
