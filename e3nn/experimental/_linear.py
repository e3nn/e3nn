import torch
from torch import nn
import numpy as np
from e3nn import o3
from e3nn.util.datatypes import Chunk


class Linearv2(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).regroup()
        irreps_out = o3.Irreps(irreps_out).regroup()
        w_ptr = 0
        paths = {}
        for (i_in, (mul_in, ir_in)), slice_in in zip(enumerate(irreps_in), irreps_in.slices()):
            for (i_out, (mul_out, ir_out)), slice_out in zip(enumerate(irreps_out), irreps_out.slices()):
                if (ir_in.l == ir_out.l) & (ir_out.p == ir_out.p):
                    paths[i_in, i_out] = (
                        Chunk(mul_in, ir_in.dim, slice_in),
                        Chunk(mul_out, ir_out.dim, slice_out),
                        (slice(w_ptr, w_ptr + mul_out * mul_in)),
                    )
                    w_ptr += mul_out * mul_in

        self.paths = paths

    def forward(
        self,
        input,
        weights,
    ):
        w_idx = 0
        chunks = []
        for (mul_in, dim, slice_in), (mul_out, dim, slice_out), (slice_weight) in self.paths.values():
            chunk_in = input[slice_in].reshape(mul_in, dim)
            weight = weights[slice_weight].reshape(mul_in, mul_out)

            norm = np.sqrt(1.0 / mul_in).astype(np.float32)

            chunk_out = torch.einsum("ui, uv -> vi", chunk_in, weight).ravel() * norm
            chunks.append(chunk_out)

            w_idx += 1

        return torch.cat(chunks)
