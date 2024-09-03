from e3nn import o3

import torch
from torch import nn


class GauntTensorProductS2Grid(nn.Module):
    """Gaunt tensor product using signals on S2."""

    # Borrowed from https://github.com/atomicarchitects/PriceofFreedom/blob/17ce493a4fbd77f14c64edb768ca00b9ef81f92f/src/tensor_products/functional.py#L210-L258
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        res_beta: int,
        res_alpha: int,
        filter_ir_out=None,
    ):
        super(GauntTensorProductS2Grid, self).__init__()
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        # Only taking in similar irreps for now
        assert irreps_in1 == irreps_in2
        if filter_ir_out is None:
            filter_ir_out = o3.s2_irreps(irreps_in1.lmax + irreps_in2.lmax)
        self.to_s2grid_in1 = o3.ToS2Grid(lmax=irreps_in1.lmax, res=(res_beta, res_alpha), fft=False)
        self.to_s2grid_in2 = o3.ToS2Grid(lmax=irreps_in2.lmax, res=(res_beta, res_alpha), fft=False)
        self.from_s2grid = o3.FromS2Grid(lmax=filter_ir_out.lmax, res=(res_beta, res_alpha), fft=False)
        self.irreps_out = o3.Irreps.spherical_harmonics(filter_ir_out.lmax)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        return self.from_s2grid(self.to_s2grid_in1(input1) * self.to_s2grid_in2(input2))
