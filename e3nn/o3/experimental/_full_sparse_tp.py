# flake8: noqa

from typing import Tuple

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
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape
        
class FullTensorProductSparse(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        leading_shape: Tuple = (),
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
        m3s = {}
        m1m2s = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(irreps_in1, irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(irreps_in2, irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l
                    cg = o3.wigner_3j(l1, l2, l3)
                    chunk = torch.zeros((2 * l3 + 1, mul_1, mul_2) + leading_shape)
                    self.register_buffer(f"chunk_{l1}_{l2}_{l3}", chunk)
                    for m3 in range(-l3, l3 + 1):
                        if (l1, l2, l3) in m3s:
                            m3s[(l1, l2, l3)].append(m3)
                        else:
                            m3s[(l1, l2, l3)] = [m3]
                        for m1 in range(-l1, l1 + 1):
                            for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                                if (m2 < -l2) or (m2 > l2):
                                    continue
                                if (l1, l2, l3, m3) in m1m2s:
                                    m1m2s[(l1, l2, l3, m3)].append((m1, m2))
                                else:
                                    m1m2s[(l1, l2, l3, m3)] = [(m1, m2)]
                                cg_coeff = cg[l1 + m1, l2 + m2, l3 + m3]
                                if irrep_normalization == "component":
                                    cg_coeff *= np.sqrt(ir_out.dim)
                                elif irrep_normalization == "norm":
                                    cg_coeff *= np.sqrt(ir_1.dim * ir_2.dim)
                                else:
                                    raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                                self.register_buffer(f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}", cg_coeff)

                    paths[(l1, l2, l3)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        self.m3s = m3s
        self.m1m2s = m1m2s
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        chunks = []
        for (l1, l2, l3), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1_t = input1[..., slice_1].reshape(
                (
                    input_dim1,
                    mul_1,
                )
                + leading_shape
            )
            x2_t = input2[..., slice_2].reshape(
                (
                    input_dim2,
                    mul_2,
                )
                + leading_shape
            )
            chunk = getattr(self, f"chunk_{l1}_{l2}_{l3}")
            for m3 in self.m3s[(l1, l2, l3)]:
                sum = 0
                for m1, m2 in list(self.m1m2s[(l1, l2, l3, m3)]):
                    cg_coeff = getattr(self, f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}")
                    path = torch.einsum("u..., v... -> uv... ", x1_t[l1 + m1, ...], x2_t[l2 + m2, ...])
                    path *= cg_coeff
                    sum += path
                chunk[l3 + m3, ...] = sum
            chunk = torch.moveaxis(chunk, 0, -1)
            chunk = torch.reshape(chunk, leading_shape + (output_mul * output_dim, ))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)
