from e3nn.util.datatypes import Path, Chunk
from typing import Union, Callable, Optional
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


class IrrepsContext:
    def __init__(self, irreps_class):
        self._Irreps = irreps_class

    def clebsch_gordan(self, ir_1, ir_2, ir_out):
        raise NotImplementedError

    def path_hash(self, ir_1, ir_2, ir_out):
        raise NotImplementedError


class O3Context(IrrepsContext):
    def __init__(self):
        super().__init__(o3.Irreps)

    def clebsch_gordan(self, ir_1, ir_2, ir_out):
        return o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)

    def path_hash(self, ir_1, ir_2, ir_out):
        return (ir_1.l, ir_2.l, ir_out.l)


class FullTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1: Union[o3.Irreps],
        irreps_in2: Union[o3.Irreps],
        *,
        filter_ir_out: Optional[Callable] = None,
        irrep_normalization: str = "component",
        regroup_output: bool = True,
    ):
        super().__init__()
        if isinstance(irreps_in1, o3.Irreps):
            self.context = O3Context()
        else:
            raise ValueError("Must be instances of o3.Irreps")

        if not isinstance(irreps_in1, type(irreps_in2)):
            raise ValueError("Both irreps_in1 and irreps_in2 must be of the same Irreps type")

        self.irrep_normalization = irrep_normalization
        self.regroup_output = regroup_output
        self.filter_ir_out = filter_ir_out

        if self.regroup_output:
            self.irreps_in1 = self.context._Irreps(irreps_in1).regroup()
            self.irreps_in2 = self.context._Irreps(irreps_in2).regroup()

        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(self.irreps_in1, self.irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(self.irreps_in2, self.irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if self.filter_ir_out and not self.filter_ir_out(ir_out):
                        continue
                    cg = self.context.clebsch_gordan(ir_1, ir_2, ir_out)
                    if self.irrep_normalization == "component":
                        cg *= np.sqrt(ir_out.dim)
                    elif self.irrep_normalization == "norm":
                        cg *= np.sqrt(ir_1.dim * ir_2.dim)
                    else:
                        raise ValueError(f"irrep_normalization={self.irrep_normalization} not supported")

                    path_hash = self.context.path_hash(ir_1, ir_2, ir_out)
                    self.register_buffer(f"cg_{path_hash}", cg)

                    paths[path_hash] = Path(
                        input_1_chunk=Chunk(mul_1, ir_1.dim, slice_1),
                        input_2_chunk=Chunk(mul_2, ir_2.dim, slice_2),
                        output_chunk=Chunk(mul_1 * mul_2, ir_out.dim, ir_out.dim),
                    )

                    irreps_out.append((mul_1 * mul_2, ir_out))

        self.paths = paths
        self.irreps_out, _, self.inv = self.context._Irreps(irreps_out).sort()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        chunks = []
        for path_hash, path in self.paths.items():
            x1 = input1[..., path.input_1_chunk.slice].reshape(
                leading_shape + (path.input_1_chunk.mul, path.input_1_chunk.dim)
            )
            x2 = input2[..., path.input_2_chunk.slice].reshape(
                leading_shape + (path.input_2_chunk.mul, path.input_2_chunk.dim)
            )
            cg = getattr(self, f"cg_{path_hash}")
            chunk = torch.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg)
            chunk = torch.reshape(chunk, chunk.shape[:-3] + (path.output_chunk.mul * path.output_chunk.dim,))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)
