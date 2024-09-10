# flake8: noqa

from typing import Tuple
from collections import defaultdict

from e3nn.util.datatypes import Path, Chunk
from e3nn import o3

import torch
from torch import nn
import numpy as np


def _prepare_inputs(input1, input2):
    dtype = torch.promote_types(input1.dtype, input2.dtype)

    input1 = input1.to(dtype=dtype)
    input2 = input2.to(dtype=dtype)

    leading_shape = torch.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    if len(leading_shape) == 0:
        leading_shape = (1,)
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape
        
class FullTensorProductSparse(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        filter_ir_out: o3.Irreps = None,
        irrep_normalization: str = "component",
        regroup_output: bool = True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L40-L135"""
        super(FullTensorProductSparse, self).__init__()

        if regroup_output:
            irreps_in1 = o3.Irreps(irreps_in1).regroup()
            irreps_in2 = o3.Irreps(irreps_in2).regroup()

        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(irreps_in1, irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(irreps_in2, irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    l1, p1, l2, p2, l3, p3 = ir_1.l, ir_1.p, ir_2.l, ir_2.p, ir_out.l, ir_out.p
                    cg = o3.wigner_3j(l1, l2, l3)
                    for m3 in range(-l3, l3 + 1):
                        for m1 in range(-l1, l1 + 1):
                            for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                                if (m2 < -l2) or (m2 > l2):
                                    continue
                                cg_coeff = cg[l1 + m1, l2 + m2, l3 + m3]
                                if irrep_normalization == "component":
                                    cg_coeff *= np.sqrt(ir_out.dim)
                                elif irrep_normalization == "norm":
                                    cg_coeff *= np.sqrt(ir_1.dim * ir_2.dim)
                                else:
                                    raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                                self.register_buffer(f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}", cg_coeff)

                    paths[(l1, p1, l2, p2, l3, p3)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

    def single_sample_forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)

        if input1.ndim != input2.ndim:
            raise ValueError(f"Inputs must have the same number of dimensions: received {input1.shape} and {input2.shape}")

        assert input1.device == input2.device, "Inputs should be on same device"

        chunks = []
        for (l1, _, l2, _, l3, _), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1  = input1[..., slice_1].reshape(leading_shape + (mul_1, input_dim1))
            x1_t = x1.permute(2,1,0) # TODO: Breaks if batch + channel dim is more than one need a fix
            x2 = input2[..., slice_2].reshape(leading_shape + (mul_2, input_dim2))
            x2_t = x2.permute(2,1,0)
            chunk = torch.zeros((2 * l3 + 1, mul_1, mul_2) + leading_shape).to(input1.device)
            for m3 in range(-l3, l3+1):
                sum = 0
                for m1 in range(-l1, l1 +1):
                    for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                        if (m2 < -l2) or (m2 > l2):
                            continue
                        cg_coeff = getattr(self, f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}")
                        path = torch.einsum("u..., v... -> uv... ", x1_t[l1 + m1, ...], x2_t[l2 + m2, ...])
                        path *= cg_coeff
                        sum += path
                chunk[l3 + m3, ...] = sum
            chunk = chunk.permute(3,1,2,0)
            chunk = torch.reshape(chunk, chunk.shape[:-3] + (output_mul * output_dim, ))
            chunks.append(chunk)

        output = torch.cat([chunks[i] for i in self.inv], dim=-1)
        # einsum shenanigans
        if leading_shape == (1,):
            output = output.squeeze(0)
        return output