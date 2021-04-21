r"""
This is an implementation of voxel convolution

>>> test()
"""
import math
import torch

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace


class Convolution(torch.nn.Module):
    r"""convolution on voxels

    Parameters
    ----------
    irreps_in : `Irreps`
        input irreps

    irreps_out : `Irreps`
        output irreps

    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``

    diameter : float
        diameter of the filter in physical units

    num_radial_basis : int
        number of radial basis functions

    steps : tuple of float
        size of the pixel in physical units
    """
    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0), **kwargs):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]

        if 'padding' not in kwargs:
            kwargs['padding'] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis='smooth_finite',
            cutoff=True,
        )
        self.register_buffer('emb', emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=lattice,
            normalize=True,
            normalization='component'
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer('sh', sh)

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

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
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum('xyzio->oixyz', kernel)
        return sc + torch.nn.functional.conv3d(x, kernel, **self.kwargs)


class LowPassFilter(torch.nn.Module):
    def __init__(self, scale, stride=1, transposed=False, steps=(1, 1, 1)):
        super().__init__()

        sigma = 0.5 * (scale ** 2 - 1)**0.5

        size = int(1 + 2 * 2.5 * sigma)
        if size % 2 == 0:
            size += 1

        r = torch.linspace(-1, 1, size)
        x = r * steps[0] / min(steps)
        x = x[x.abs() <= 1]
        y = r * steps[1] / min(steps)
        y = y[y.abs() <= 1]
        z = r * steps[2] / min(steps)
        z = z[z.abs() <= 1]
        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        lattice = (size // 2) * lattice

        kernel = torch.exp(-lattice.norm(dim=-1).pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        if transposed:
            kernel = kernel * stride**3
        kernel = kernel[None, None]
        self.register_buffer('kernel', kernel)

        self.scale = scale
        self.stride = stride
        self.size = size
        self.transposed = transposed

    def forward(self, image):
        """
        Parameters
        ----------
        image : `torch.Tensor`
            tensor of shape ``(..., x, y, z)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., x, y, z)``
        """
        if self.scale <= 1:
            assert self.stride == 1
            return image

        out = image
        out = out.reshape(-1, 1, *out.shape[-3:])
        if self.transposed:
            out = torch.nn.functional.conv_transpose3d(out, self.kernel, padding=self.size // 2, stride=self.stride)
        else:
            out = torch.nn.functional.conv3d(out, self.kernel, padding=self.size // 2, stride=self.stride)
        out = out.reshape(*image.shape[:-3], *out.shape[-3:])
        return out


def test():
    conv = Convolution("0e + 1e", "0e + 1e + 1o + 2e + 2o", o3.Irreps.spherical_harmonics(lmax=3), diameter=5, num_radial_basis=5, steps=(1, 1, 1))

    x = torch.randn(10, 4, 32, 32, 32)
    conv(x)

    fi = LowPassFilter(2.0)

    fi(x)
