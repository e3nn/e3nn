import math

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import FullyConnectedTensorProduct, Linear

try:
    from MinkowskiEngine import KernelGenerator, MinkowskiConvolutionFunction, SparseTensor
    from MinkowskiEngineBackend._C import ConvolutionMode
except ImportError:
    pass


class Convolution(torch.nn.Module):
    r"""convolution on voxels

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        input irreps

    irreps_out : `e3nn.o3.Irreps`
        output irreps

    irreps_sh : `e3nn.o3.Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``

    diameter : float
        diameter of the filter in physical units

    num_radial_basis : int
        number of radial basis functions

    steps : tuple of float
        size of the pixel in physical units
    """
    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0)):
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

        lattice = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)  # [x, y, z, R^3]
        self.register_buffer('lattice', lattice)

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

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out,
            shared_weights=False,
            compile_left_right=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

        self.kernel_generator = KernelGenerator(lattice.shape[:3], dimension=3)

        self.conv_fn = MinkowskiConvolutionFunction()

    def kernel(self):
        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]

        # TODO: understand why this is necessary
        kernel = torch.einsum('xyzij->zyxij', kernel)  # [z, y, x, irreps_in.dim, irreps_out.dim]

        kernel = kernel.reshape(-1, *kernel.shape[-2:])  # [z * y * x, irreps_in.dim, irreps_out.dim]
        return kernel

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : SparseTensor

        Returns
        -------
        SparseTensor
        """
        sc = self.sc(x.F)

        out = self.conv_fn.apply(
            x.F,
            self.kernel(),
            self.kernel_generator,
            ConvolutionMode.DEFAULT,
            x.coordinate_map_key,
            x.coordinate_map_key,
            x._manager,
        )

        return SparseTensor(
            sc + out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x._manager,
        )
