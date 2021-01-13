r"""
This is a tentative implementation of voxel convolution

>>> test()
"""
import math
import torch

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace


class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, size):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=3)
        self.size = size
        self.num_gaussian = self.size

        # self-interaction
        self.si = Linear(self.irreps_in, self.irreps_out)

        # interaction with neighbors
        r = torch.linspace(-1, 1, self.size)
        lattice = torch.stack(torch.meshgrid(r, r, r), dim=-1)
        self.register_buffer('d', lattice.norm(dim=-1))

        sh = o3.spherical_harmonics(self.irreps_sh, lattice, True, 'component')  # [x, y, z, irreps_sh.dim]
        self.register_buffer('sh', sh)

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)

        self.weight = torch.nn.Parameter(torch.randn(self.num_gaussian, self.tp.weight_numel))

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``

        """
        si = self.si(x.transpose(1, 4))
        si = si.transpose(1, 4)

        weight = soft_one_hot_linspace(self.d, 0.0, 1.0, self.num_gaussian) @ self.weight
        weight = weight * (math.pi * self.d).cos()[:, :, :, None] / (self.size ** (3/2))
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum('xyzio->oixyz', kernel)
        return si + 0.1 * torch.nn.functional.conv3d(x, kernel, padding=self.size // 2)


def test():
    conv = Convolution("0e + 1e", "0e + 1e + 1o + 2e + 2o", 5)

    x = torch.randn(10, 4, 32, 32, 32)
    conv(x)
