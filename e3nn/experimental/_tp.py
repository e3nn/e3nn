# from math import sqrt
# from typing import List, Optional, Union, Any, Callable
# import warnings

# import torch
# from torch import nn

# from e3nn import o3, experimental
# from e3nn.o3._tensor_product._codegen import _sum_tensors
# from e3nn.util import prod
# from e3nn.util.datatypes import Instruction, Chunk
# from ._basic import from_chunks


# def _flat_concatenate(xs):
#     if any(x is None for x in xs):
#         return None
#     if len(xs) > 0:
#         return torch.cat([x.flatten() for x in xs])
#     return torch.zeros((0,), dtype=torch.float32)


# class TensorProduct(nn.Module):
#     def __init__(
#         self,
#         irreps_in1: o3.Irreps,
#         irreps_in2: o3.Irreps,
#         irreps_out: o3.Irreps,
#         instructions: List[tuple],
#         in1_var: Optional[Union[List[float], torch.Tensor]] = None,
#         in2_var: Optional[Union[List[float], torch.Tensor]] = None,
#         out_var: Optional[Union[List[float], torch.Tensor]] = None,
#         irrep_normalization: str = None,
#         path_normalization: str = None,
#     ) -> None:
        
#         pass
    
#     #     super(TensorProduct, self).__init__()

#     #     self.irreps_in1 = o3.Irreps(irreps_in1)
#     #     self.irreps_in1_slices = self.irreps_in1.slices()
#     #     self.irreps_in2 = o3.Irreps(irreps_in2)
#     #     self.irreps_in2_slices = self.irreps_in2.slices()
#     #     self.irreps_out = o3.Irreps(irreps_out)
#     #     self.irreps_out_slices = self.irreps_out.slices()

#     #     del irreps_in1, irreps_in2, irreps_out

#     #     if irrep_normalization is None:
#     #         irrep_normalization = "component"

#     #     if path_normalization is None:
#     #         path_normalization = "element"

#     #     assert irrep_normalization in ["component", "norm", "none"]
#     #     assert path_normalization in ["element", "path", "none"]

#     #     instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]

#     #     instructions = [
#     #         Instruction(
#     #             i_in1,
#     #             i_in2,
#     #             i_out,
#     #             connection_mode,
#     #             has_weight,
#     #             path_weight,
#     #             None,
#     #             Chunk(self.irreps_in1[i_in1].mul, self.irreps_in1[i_in1].ir.dim, self.irreps_in1_slices[i_in1]),
#     #             Chunk(self.irreps_in2[i_in2].mul, self.irreps_in2[i_in2].ir.dim, self.irreps_in2_slices[i_in2]),
#     #             Chunk(self.irreps_out[i_out].mul, self.irreps_out[i_out].ir.dim, self.irreps_out_slices[i_out]),
#     #         )
#     #         for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
#     #     ]

#     #     if in1_var is None:
#     #         in1_var = [1.0 for _ in self.irreps_in1]

#     #     if in2_var is None:
#     #         in2_var = [1.0 for _ in self.irreps_in2]

#     #     if out_var is None:
#     #         out_var = [1.0 for _ in self.irreps_out]

#     #     def num_elements(ins):
#     #         return {
#     #             "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
#     #             "uvu": self.irreps_in2[ins.i_in2].mul,
#     #             "uvv": self.irreps_in1[ins.i_in1].mul,
#     #             "uuw": self.irreps_in1[ins.i_in1].mul,
#     #             "uuu": 1,
#     #             "uvuv": 1,
#     #             "uvu<v": 1,
#     #             "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
#     #         }[ins.connection_mode]

#     #     normalization_coefficients = []

#     #     for ins in instructions:
#     #         mul_ir_in1 = self.irreps_in1[ins.i_in1]
#     #         mul_ir_in2 = self.irreps_in2[ins.i_in2]
#     #         mul_ir_out = self.irreps_out[ins.i_out]
#     #         assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
#     #         assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
#     #         assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

#     #         if irrep_normalization == "component":
#     #             alpha = mul_ir_out.ir.dim
#     #         if irrep_normalization == "norm":
#     #             alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
#     #         if irrep_normalization == "none":
#     #             alpha = 1

#     #         if path_normalization == "element":
#     #             x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i) for i in instructions if i.i_out == ins.i_out)
#     #         if path_normalization == "path":
#     #             x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
#     #             x *= len([i for i in instructions if i.i_out == ins.i_out])
#     #         if path_normalization == "none":
#     #             x = 1

#     #         if x > 0.0:
#     #             alpha /= x

#     #         alpha *= out_var[ins.i_out]
#     #         alpha *= ins.path_weight

#     #         normalization_coefficients += [sqrt(alpha)]

#     #     self.instructions = [
#     #         Instruction(
#     #             ins.i_in1,
#     #             ins.i_in2,
#     #             ins.i_out,
#     #             ins.connection_mode,
#     #             ins.has_weight,
#     #             alpha,
#     #             ins.path_shape,
#     #             ins.chunk_in1,
#     #             ins.chunk_in2,
#     #             ins.chunk_out,
#     #         )
#     #         for ins, alpha in zip(instructions, normalization_coefficients)
#     #     ]

#     #     if self.irreps_out.dim > 0:
#     #         output_mask = torch.cat(
#     #             [
#     #                 torch.ones(mul * ir.dim)
#     #                 if any(
#     #                     (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
#     #                     for ins in self.instructions
#     #                 )
#     #                 else torch.zeros(mul * ir.dim)
#     #                 for i_out, (mul, ir) in enumerate(self.irreps_out)
#     #             ]
#     #         )
#     #     else:
#     #         output_mask = torch.ones(0)
#     #     self.register_buffer("output_mask", output_mask)

#     #     self.irreps_out, _, self.inv = self.irreps_out.sort()

#     # def forward(self, x1, x2, weights: Union[List[torch.Tensor], torch.Tensor]):
#     #     if x2 is None:
#     #         w, x1, x2 = [], w, x1

#     #     if isinstance(weights, list):
#     #         assert len(weights) == len([ins for ins in self.instructions if ins.has_weight]), (
#     #             len(weights),
#     #             len([ins for ins in self.instructions if ins.has_weight]),
#     #         )
#     #         weights_flat = _flat_concatenate(weights)
#     #         weights_list = weights
#     #     else:
#     #         weights_flat = weights
#     #         weights_list = []
#     #         i = 0
#     #         for ins in self.instructions:
#     #             if ins.has_weight:
#     #                 n = prod(ins.path_shape)
#     #                 weights_list.append(weights[i : i + n].reshape(ins.path_shape))
#     #                 i += n
#     #         assert i == weights.size
#     #     del weights

#     #     assert x1.ndim == 1, f"input1 is shape {x1.shape}. Execting ndim to be 1. Use torch.vmap to map over input1"
#     #     assert x1.ndim == 1, f"input2 is shape {x2.shape}. Execting ndim to be 1. Use torch.vmap to map over input2"

#     #     if len(self.instructions) == 0:
#     #         output = x.new_zeros(output_shape + (self.irreps_out.dim,))
#     #         return output

#     #     # = extract individual input irreps =
#     #     # If only one input irrep, can avoid creating a view
#     #     if len(self.irreps_in1) == 1:
#     #         x1_list = [x1.reshape(self.irreps_in1[0].mul, self.irreps_in1[0].ir.dim)]
#     #     else:
#     #         x1_list = [
#     #             (x1[:, i] if x1.ndim > 1 else x1[i]).reshape(mul_ir.mul, mul_ir.ir.dim)
#     #             for i, mul_ir in zip(self.irreps_in1.slices(), self.irreps_in1)
#     #         ]

#     #     x2_list = []
#     #     # If only one input irrep, can avoid creating a view
#     #     if len(self.irreps_in2) == 1:
#     #         x2_list.append(x2.reshape(self.irreps_in2[0].mul, self.irreps_in2[0].ir.dim))
#     #     else:
#     #         x2_list = [
#     #             (x2[:, i] if x2.ndim > 1 else x2[i]).reshape(mul_ir.mul, mul_ir.ir.dim)
#     #             for i, mul_ir in zip(self.irreps_in2.slices(), self.irreps_in2)
#     #         ]

#     #     # Current index in the flat weight tensor
#     #     weight_index = 0

#     #     outputs = []

#     #     for ins in self.instructions:
#     #         mul_ir_in1 = self.irreps_in1[ins.i_in1]
#     #         mul_ir_in2 = self.irreps_in2[ins.i_in2]
#     #         mul_ir_out = self.irreps_out[ins.i_out]

#     #         assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
#     #         assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

#     #         if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
#     #             continue

#     #         x1 = x1_list[ins.i_in1]
#     #         x2 = x2_list[ins.i_in2]

#     #         assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

#     #         if ins.has_weight:
#     #             w = weights[weight_index]
#     #             assert w.shape == ins.path_shape
#     #             weight_index += 1

#     #         w3j = getattr(self, f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}")

#     #         if ins.connection_mode == "uvw":
#     #             assert ins.has_weight
#     #             result = torch.einsum("uvw,ijk,ui,vj->wk", w, w3j, x1, x2)
#     #         if ins.connection_mode == "uvu":
#     #             assert mul_ir_in1.mul == mul_ir_out.mul
#     #             if ins.has_weight:
#     #                 result = torch.einsum("uv,ijk,ui,vj->uk", w, w3j, x1, x2)
#     #             else:
#     #                 # not so useful operation because v is summed
#     #                 result = torch.einsum("ijk,ui,vj->uk", w3j, x, y)
#     #         if ins.connection_mode == "uvv":
#     #             assert mul_ir_in2.mul == mul_ir_out.mul
#     #             if ins.has_weight:
#     #                 result = torch.einsum("uv,ijk,ui,vj->vk", w, w3j, x1, x2)
#     #             else:
#     #                 # not so useful operation because u is summed
#     #                 result = torch.einsum("ijk,ui,vj->vk", w3j, x1, x2)
#     #         if ins.connection_mode == "uuw":
#     #             assert mul_ir_in1.mul == mul_ir_in2.mul
#     #             if ins.has_weight:
#     #                 result = torch.einsum("uw,ijk,ui,uj->zwk", w, w3j, x1, x2)
#     #             else:
#     #                 # equivalent to tp(x, y, 'uuu').sum('u')
#     #                 assert mul_ir_out.mul == 1
#     #                 result = torch.einsum("ijk,ui,uj->k", w3j, x1, x2)
#     #         if ins.connection_mode == "uuu":
#     #             assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
#     #             if ins.has_weight:
#     #                 result = torch.einsum("u,ijk,ui,uj->uk", w, w3j, x1, x2)
#     #             else:
#     #                 result = torch.einsum("ijk,ui,uj->uk", w3j, x1, x2)
#     #         if ins.connection_mode == "uvuv":
#     #             assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
#     #             if ins.has_weight:
#     #                 # TODO implement specialized code
#     #                 result = torch.einsum("uv,ijk,ui,vj->uvk", w, w3j, x1, x2)
#     #             else:
#     #                 # TODO implement specialized code
#     #                 result = torch.einsum("ijk,ui,vj->uvk", w3j, x1, x2)
#     #         if ins.connection_mode == "uvu<v":
#     #             assert mul_ir_in1.mul == mul_ir_in2.mul
#     #             assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
#     #             i = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
#     #             xx = torch.einsum("ui,vj->uvij", x1, x2)[i[0], i[1]]  # uvij -> wij
#     #             if ins.has_weight:
#     #                 # TODO implement specialized code
#     #                 result = torch.einsum("w,ijk,wij->wk", w, w3j, xx)
#     #             else:
#     #                 # TODO implement specialized code
#     #                 result = torch.einsum("ijk,wij->wk", w3j, xx)
#     #         if ins.connection_mode == "u<vw":
#     #             assert mul_ir_in1.mul == mul_ir_in2.mul
#     #             assert ins.has_weight
#     #             i = torch.triu_indices(mul_ir_in1.mul, 1)
#     #             xx = xx[:, i[0], i[1]]  # zuvij -> zqij
#     #             # TODO implement specialized code
#     #             result = torch.einsum("qw,ijk,qij->wk", w, w3j, torch.einsum("ui,vj->uvij", x, y)[i[0], i[1]])

#     #         outputs += [result]

#     #     out = [
#     #         _sum_tensors(
#     #             [out for ins, out in zip(self.instructions, outputs) if ins.i_out == i_out],
#     #             shape=(mul_ir_out.mul, mul_ir_out.ir.dim),
#     #             like=x1,
#     #         )
#     #         for i_out, mul_ir_out in enumerate(self.irreps_out)
#     #     ]

#     #     return from_chunks(out, self.inv)
