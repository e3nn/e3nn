# flake8: noqa

from e3nn.util.datatypes import Path, Chunk, TensorProductMode
from e3nn import o3

import torch
from torch import nn
import numpy as np

import itertools


class TensorSquare(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        *,
        irrep_normalization: str = "component",
        normalized_input: bool = False,
        regroup_output: bool = True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L216-L372"""
        super(TensorSquare, self).__init__()

        if regroup_output:
            irreps_in = o3.Irreps(irreps_in).regroup()

        paths = {}
        irreps_out = []
        for i_1, ((mul_1, ir_1), slice_1) in enumerate(zip(irreps_in, irreps_in.slices())):
            for i_2, ((mul_2, ir_2), slice_2) in enumerate(zip(irreps_in, irreps_in.slices())):
                for ir_out in ir_1 * ir_2:
                    if normalized_input:
                        if irrep_normalization == "component":
                            alpha = ir_1.dim * ir_2.dim * ir_out.dim
                        elif irrep_normalization == "norm":
                            alpha = ir_1.dim * ir_2.dim
                        elif irrep_normalization == "none":
                            alpha = 1
                        else:
                            raise ValueError(f"irrep_normalization={irrep_normalization}")
                    else:
                        if irrep_normalization == "component":
                            alpha = ir_out.dim
                        elif irrep_normalization == "norm":
                            alpha = ir_1.dim * ir_2.dim
                        elif irrep_normalization == "none":
                            alpha = 1
                        else:
                            raise ValueError(f"irrep_normalization={irrep_normalization}")

                    if i_1 < i_2:
                        paths[(ir_1.l, ir_1.p, ir_2.l, ir_2.p, ir_out.l, ir_out.p)] = Path(
                            Chunk(mul_1, ir_1.dim, slice_1),
                            Chunk(mul_2, ir_2.dim, slice_2),
                            Chunk(mul_1 * mul_2, ir_out.dim),
                            tensor_product_mode=TensorProductMode.UVW
                        )
                        irreps_out.append((mul_1 * mul_2, ir_out))
                    elif i_1 == i_2:
                        if mul_1 > 1:
                            irreps_out.append((mul_1 * (mul_1 - 1) // 2, ir_out))
                            uvw = torch.zeros((mul_1, mul_1, mul_1 * (mul_1 - 1) // 2))
                            i, j = zip(*itertools.combinations(range(mul_1), 2))
                            uvw[i, j, torch.arange(len(i))] = 1
                            paths[(ir_1.l, ir_1.p, ir_2.l, ir_2.p, ir_out.l, ir_out.p)] = Path(
                                input_1_slice=Chunk(mul_1, ir_1.dim, slice_1),
                                output_slice=Chunk(int(uvw.shape[-1]), ir_out.dim),
                                tensor_product_mode=TensorProductMode.UVUV
                            )
                            self.register_buffer(f"uvw_{ir_1.l}_{ir_2.l}_{ir_out.l}", uvw)

                        if ir_out.l % 2 == 0:
                            if normalized_input:
                                if irrep_normalization == "component":
                                    if ir_out.l == 0:
                                        alpha = ir_out.dim * ir_1.dim
                                    else:
                                        alpha = ir_1.dim * (ir_1.dim + 2) / 2 * ir_out.dim
                                elif irrep_normalization == "norm":
                                    if ir_out.l == 0:
                                        alpha = ir_out.dim * ir_1.dim
                                    else:
                                        alpha = ir_1.dim * (ir_1.dim + 2) / 2
                                elif irrep_normalization == "none":
                                    alpha = 1
                                else:
                                    raise ValueError(f"irrep_normalization={irrep_normalization}")
                            else:
                                if irrep_normalization == "component":
                                    if ir_out.l == 0:
                                        alpha = ir_out.dim / (ir_1.dim + 2)
                                    else:
                                        alpha = ir_out.dim / 2
                                elif irrep_normalization == "norm":
                                    if ir_out.l == 0:
                                        alpha = ir_1.dim * ir_2.dim / (ir_1.dim + 2)
                                    else:
                                        alpha = ir_1.dim * ir_2.dim / 2
                                elif irrep_normalization == "none":
                                    alpha = 1
                                else:
                                    raise ValueError(f"irrep_normalization={irrep_normalization}")
                            irreps_out.append((mul_1, ir_out))
                            paths[(ir_1.l, ir_1.p, ir_2.l, ir_2.p, ir_out.l, ir_out.p)] = Path(
                                input_1_slice=Chunk(mul_1, ir_1.dim, slice_1),
                                output_slice=Chunk(mul_1, ir_out.dim),
                                tensor_product_mode=TensorProductMode.UUU
                            )

                    cg = o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)
                    cg *= np.sqrt(alpha)
                    self.register_buffer(f"cg_{ir_1.l}_{ir_2.l}_{ir_out.l}", cg)

        self.paths = paths
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.irreps_in = irreps_in

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        for (l1, _, l2, _, l3, _), path in self.paths.items():
            match path.tensor_product_mode:
                case TensorProductMode.UVW:
                    ((mul_1, input_dim1, slice_1),
                     (mul_2, input_dim2, slice_2),
                     (output_mul, output_dim, _), (_)) = path
                    x1 = input[..., slice_1].reshape(-1, mul_1, input_dim1)
                    x2 = input[..., slice_2].reshape(-1, mul_2, input_dim2)
                    cg = getattr(self, f"cg_{l1}_{l2}_{l3}")
                    chunk = torch.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg)
                    chunk = torch.reshape(chunk, chunk.shape[:-3] + (output_mul * output_dim,))
                    chunks.append(chunk)

                case TensorProductMode.UVUV:
                    ((mul_in, input_dim, slice_in), _, (output_mul, output_dim, _), (_)) = path
                    x = input[..., slice_in].reshape(-1, mul_in, input_dim)
                    cg = getattr(self, f"cg_{l1}_{l2}_{l3}")
                    uvw = getattr(self, f"uvw_{l1}_{l2}_{l3}")
                    chunk = torch.einsum("...ui, ...vj, ijk, uvw -> ...wk", x, x, cg, uvw)
                    chunk = torch.reshape(chunk, chunk.shape[:-2] + (output_mul * output_dim,))
                    chunks.append(chunk)

                case TensorProductMode.UUU:
                    ((mul_in, input_dim, slice_in), _, (output_mul, output_dim, _), (_)) = path
                    x = input[..., slice_in].reshape(-1, mul_in, input_dim)
                    cg = getattr(self, f"cg_{l1}_{l2}_{l3}")
                    chunk = torch.einsum("...ui, ...uj, ijk -> ...uk", x, x, cg)
                    chunk = torch.reshape(chunk, chunk.shape[:-2] + (output_mul * output_dim,))
                    chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)