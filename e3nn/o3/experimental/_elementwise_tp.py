# flake8: noqa

from e3nn.util.datatypes import Path, Chunk
from e3nn import o3

import torch
from torch import nn
import numpy as np
from ._full_tp import _prepare_inputs


class ElementwiseTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        filter_ir_out: o3.Irreps = None,
        irrep_normalization: str = "component",
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L139-L213"""
        super(ElementwiseTensorProduct, self).__init__()

        if irreps_in1.num_irreps != irreps_in2.num_irreps:
            raise ValueError(
                "o3.ElementwiseTensorProductv2: inputs must have the same number of irreps, "
                f"got {irreps_in1.num_irreps} and {irreps_in2.num_irreps}"
            )

        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1, (mul_2, ir_2), slice_2 in zip(
            irreps_in1, irreps_in1.slices(), irreps_in2, irreps_in2.slices()
        ):
            for ir_out in ir_1 * ir_2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue
                cg = o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)
                if irrep_normalization == "component":
                    cg *= np.sqrt(ir_out.dim)
                elif irrep_normalization == "norm":
                    cg *= np.sqrt(ir_1.dim * ir_2.dim)
                else:
                    raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                self.register_buffer(f"cg_{ir_1.l}_{ir_2.l}_{ir_out.l}", cg)
                paths[(ir_1.l, ir_2.l, ir_out.l)] = Path(
                    Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_1, ir_2.dim, slice_2), Chunk(mul_1, ir_out.dim)
                )
                irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
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

        # Assumes that input1 and input2 are aligned

        chunks = []
        for (l1, l2, l3), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1 = input1[..., slice_1].reshape(
                leading_shape
                + (
                    mul_1,
                    input_dim1,
                )
            )
            x2 = input2[..., slice_2].reshape(
                leading_shape
                + (
                    mul_2,
                    input_dim2,
                )
            )
            cg = getattr(self, f"cg_{l1}_{l2}_{l3}")
            chunk = torch.einsum("...ui, ...uj, ijk -> ...uk", x1, x2, cg)
            chunk = torch.reshape(chunk, chunk.shape[:-2] + (output_mul * output_dim,))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)
