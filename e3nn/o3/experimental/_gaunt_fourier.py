from e3nn import o3

import torch
from torch import nn

from . import _gtp_utils as gtp_utils


class GauntTensorProductFourier2D(nn.Module):
    "Borrowed from https://github.com/atomicarchitects/PriceofFreedom/blob/17ce493a4fbd77f14c64edb768ca00b9ef81f92f/src/tensor_products/functional.py#L158-L207"

    def __init__(self, irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, res_theta: int, res_phi: int, convolution_type: str):
        super(GauntTensorProductFourier2D, self).__init__()
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        # Only taking in similar irreps for now
        assert irreps_in1 == irreps_in2

        self.y1_grid = gtp_utils.compute_y_grid(irreps_in1.lmax, res_theta=res_theta, res_phi=res_phi)
        self.y2_grid = gtp_utils.compute_y_grid(irreps_in2.lmax, res_theta=res_theta, res_phi=res_phi)
        self.z_grid = gtp_utils.compute_z_grid(irreps_in1.lmax + irreps_in2.lmax, res_theta=res_theta, res_phi=res_phi)
        self.irreps_out = o3.Irreps.spherical_harmonics(irreps_in1.lmax + irreps_in2.lmax)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        # Convert to 2D Fourier coefficients.
        input1_uv = torch.einsum("...a,auv->...uv", input1, self.y1_grid)
        input2_uv = torch.einsum("...a,auv->...uv", input2, self.y2_grid)

        # Perform the convolution in Fourier space, either directly or using FFT.
        if self.convolution_type == "direct":
            output_uv = gtp_utils.convolve_2D_direct(input1_uv, input2_uv)
        elif self.convolution_type == "fft":
            output_uv = gtp_utils.convolve_2D_fft(input1_uv, input2_uv)
        else:
            raise ValueError(f"Unknown convolution type {self.convolution_type}.")

        # Convert back to Real SH coefficients.
        output_lm = torch.einsum("...uv,auv->...a", output_uv.conj(), self.z_grid).real

        return output_lm
