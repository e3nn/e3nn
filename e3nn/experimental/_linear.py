from math import sqrt
from typing import Callable, NamedTuple, Optional, Tuple, Union, List, Any

import numpy as np

import e3nn
from e3nn import o3
from e3nn.o3._tensor_product._codegen import _sum_tensors
from e3nn.util import prod
from e3nn.util.datatypes import Instruction, Chunk

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        channel_in: Optional[int] = None,
        channel_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Optional[Union[List[bool], bool]] = None,
        path_normalization: str = "element",
    ):
        super(Linear, self).__init__()

        assert path_normalization in ["element", "path"]
        # if path_normalization is None:
        #     path_normalization = config("path_normalization")
        if isinstance(path_normalization, str):
            path_normalization = {"element": 0.0, "path": 1.0}[path_normalization]

        # if gradient_normalization is None:
        #     gradient_normalization = config("gradient_normalization")
        # if isinstance(gradient_normalization, str):
        #     gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_in_slices = irreps_in.slices()
        irreps_out = o3.Irreps(irreps_out).simplify()
        irreps_out_slices = irreps_out.slices()

        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                chunk_in=Chunk(irreps_in[i_in].mul, irreps_in[i_in].ir.dim, irreps_in_slices[i_in]),
                i_out=i_out,
                chunk_out=Chunk(irreps_out[i_in].mul, irreps_out[i_out].ir.dim, irreps_out_slices[i_out]),
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for (i_in, i_out) in instructions
        ]

        def alpha(this):
            x = irreps_in[this.i_in].mul ** path_normalization * sum(
                irreps_in[other.i_in].mul ** (1.0 - path_normalization) for other in instructions if other.i_out == this.i_out
            )
            return 1 / x if x > 0 else 1.0

        instructions = [
            Instruction(
                i_in=ins.i_in,
                chunk_in=ins.chunk_in,
                i_out=ins.i_out,
                chunk_out=ins.chunk_out,
                path_shape=ins.path_shape,
                path_weight=sqrt(alpha(ins)) ** (-0.5),
            )
            for ins in instructions
        ]
        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir):
                raise ValueError(f"{ins.i_in} and {ins.i_out} do not have the same irrep")

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(
                i_in=-1,
                chunk_in=irreps_in_slices[-1],
                i_out=i_out,
                chunk_out=irreps_out_slices[i_out],
                path_shape=(mul_ir.dim,),
                path_weight=1.0,
            )
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        # == Process arguments ==
        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)
        self.weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1)

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(
                torch.ones(*((channel_in, channel_out) if channel_in is not None else ()), self.weight_numel)
            )

        # == Generate biases ==
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = torch.nn.Parameter(
                torch.zeros(*((channel_out,) if channel_out is not None else ()), self.bias_numel)
            )  # see appendix C.1 and Eq.5 of https://arxiv.org/pdf/2011.14522.pdf

        if irreps_out.dim > 0:
            output_mask = torch.concatenate(
                [
                    (
                        torch.randn(mul_ir.dim)
                        if any((ins.i_out == i_out) and (0 not in ins.path_shape) for ins in instructions)
                        else torch.zeros(mul_ir.dim)
                    )
                    for i_out, mul_ir in enumerate(irreps_out)
                ]
            )
        else:
            output_mask = torch.ones(0, bool)

        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.irreps_out_dim = self.irreps_out.dim
        self.instructions = instructions
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.register_buffer("output_mask", output_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(self, features, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias

        if self.channel_in is None:
            size = input.shape[:-1]
            outsize = size + (self.irreps_out_dim,)
        else:
            size = input.shape[:-2]
            outsize = size + (
                self.channel_out,
                self.irreps_out_dim,
            )

        if len(self.instructions) == 0 and self.bias_numel == 0:
            return input.new_zeros(outsize)

        if self.channel_in is None:
            input = input.reshape(-1, self._irreps_in_dim)
        else:
            input = input.reshape(-1, self.channel_in, self._irreps_in_dim)

        batch_out = input.shape[0]
        if self.weight_numel > 0:
            ws = (
                ws.reshape(-1, self.weight_numel)
                if self.channel_in is None
                else ws.reshape(-1, self.channel_in, self.channel_out, self.weight_numel)
            )

        if self.bias_numel > 0:
            if self.channel_out is None:
                bs = bs.reshape(-1, self.bias_numel)
            else:
                bs = bs.reshape(-1, self.channel_out, self.bias_numel)

        # = extract individual input irreps =
        if len(self.instructionns) == 1:
            input_list = [
                input.reshape(
                    batch_out, *(() if self.channel_in is None else (self.channel_in,)), ins.chunk_in.mul, ins.chunk_in.dim
                )
            ]
        else:
            input_list = [input.narrow(-1, ins.chunk_in.slice.start, ins.chunk_in.dim) for ins in self.instructions]

        z = "" if self.shared_weights else "z"

        flat_weight_index = 0
        flat_bias_index = 0
        out_list = []

        for ins in self.instructions:
            mul_ir_out = ins.chunk_out.mul

            if ins.i_in == -1:
                # = bias =
                b = bs.narrow(-1, flat_bias_index, prod(ins.path_shape))
                flat_bias_index += prod(ins.path_shape)
                out_list += [
                    (ins.path_weight * b).reshape(
                        1, *(() if self.channel_out is None else (self.channel_out,)), mul_ir_out.dim
                    )
                ]
            else:
                mul_ir_in = self.chunk_in.mul

                # Short-circut for empty irreps
                if ins.chunk_in.dim == 0 or ins.chunk_out.dim == 0:
                    continue

                # Extract the weight from the flattened weight tensor
                path_nweight = prod(ins.path_shape)
                if len(self.instructions) == 1:
                    # Avoid unnecessary view when there is only one weight
                    w = ws
                else:
                    w = ws.narrow(-1, flat_weight_index, path_nweight)
                w = w.reshape(
                    (() if self.shared_weights else (-1,))
                    + (() if self.channel_in is None else (self.channel_in, self.channel_out))
                    + ins.path_shape
                )
                flat_weight_index += path_nweight
                if self.channel_in is None:
                    ein_out = torch.einsum(f"{z}uw,zui->zwi", w, input_list[ins.i_in])

                else:
                    ein_out = torch.einsum(f"{z}xyuw,zxui->zywi", w, input_list[ins.i_in])

                ein_out = ins.path_weight * ein_out
                out_list += [
                    ein_out.reshape(batch_out, *(() if self.channel_out is None else (self.channel_out,)), mul_ir_out.dim)
                ]

        # = Return the result =
        out = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(batch_out, *(() if self.channel_out is None else (self.channel_out,)), ins.chunk_out.dim),
                empty_return_none=True,
            )
            for i_out, _ in enumerate(self.irreps_out)
            if ins.chunk_out.mul > 0
        ]
        if len(out) > 1:
            out = torch.cat(out, dim=-1)
        else:
            out = out[0]

        out = out.reshape(outsize)
        return out

    # @property
    # def num_weights(self) -> int:
    #     return sum(np.prod(i.path_shape) for i in self.instructions)

    # def split_weights(self, weights: torch.Tensor) -> List[torch.Tensor]:
    #     # This code is not functional
    #     ws = []
    #     cursor = 0
    #     for i in self.instructions:
    #         ws += [weights[cursor : cursor + np.prod(i.path_shape)].reshape(i.path_shape)]
    #         cursor += np.prod(i.path_shape)
    #     return ws

    # def matrix(self, ws: List[torch.Tensor]) -> torch.Tensor:
    #     r"""Compute the matrix representation of the linear operator.

    #     Args:
    #         ws: List of weights.

    #     Returns:
    #         The matrix representation of the linear operator. The matrix is shape ``(irreps_in.dim, irreps_out.dim)``.
    #     """
    #     dtype = ws[0].dtype  # Assuming that all ws have same dtype
    #     output = torch.zeros((self.irreps_in.dim, self.irreps_out.dim), dtype)
    #     for ins, w in zip(self.instructions, ws):
    #         assert ins.i_in != -1
    #         mul_in, ir_in = self.irreps_in[ins.i_in]
    #         mul_out, ir_out = self.irreps_out[ins.i_out]
    #         output[self.irreps_in.slices()[ins.i_in], self.irreps_out.slices()[ins.i_out]] += ins.path_weight * torch.einsum(
    #             "uw,ij->uiwj", w, torch.eye(ir_in.dim, dtype=dtype)
    #         ).reshape((mul_in * ir_in.dim, mul_out * ir_out.dim))
    #     return output

    # def __repr__(self):
    #     return (
    #         f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out}, "
    #         f"{len(self.instructions)} instructions, {self.num_weights} weights)"
    #     )


# def _get_gradient_normalization(gradient_normalization: Optional[Union[float, str]]) -> float:
#     """Get the gradient normalization from the config or from the argument."""
#     if gradient_normalization is None:
#         gradient_normalization = config("gradient_normalization")
#     if isinstance(gradient_normalization, str):
#         return {"element": 0.0, "path": 1.0}[gradient_normalization]
#     return gradient_normalization


# class Linear(nn.Module):
#     r"""Equivariant Linear torch module

#     Args:
#         irreps_out (`Irreps`): output representations, if allowed bu Schur's lemma.
#         channel_out (optional int): if specified, the last axis before the irreps
#             is assumed to be the channel axis and is mixed with the irreps.
#         irreps_in (`Irreps`): input representations. If not specified,
#             the input representations is obtained when calling the module.
#         channel_in (optional int): required when using 'mixed_per_channel' linear_type,
#             indicating the size of the last axis before the irreps in the input.
#         biases (bool): whether to add a bias to the output.
#         path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
#             0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
#         gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
#             0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
#         num_indexed_weights (optional int): number of indexed weights. See example below.
#         weights_per_channel (bool): whether to have one set of weights per channel.
#         force_irreps_out (bool): whether to force the output irreps to be the one specified in ``irreps_out``.

#     Due to how nn.Module is implemented, irreps_in and irreps_out must be supplied at initialization.
#     The type of the linear layer must also be supplied at initialization:
#     'vanilla', 'indexed', 'mixed', 'mixed_per_channel'
#     Also, depending on what type of linear layer is used, additional options
#     (eg. 'num_indexed_weights', 'weights_per_channel', 'weights_dim', 'channel_in')
#     must be supplied.
#     """

#     def __init__(
#         self,
#         irreps_in: o3.Irreps,
#         irreps_out: o3.Irreps,
#         *,
#         channel_out: Optional[int] = None,
#         channel_in: Optional[int] = None,
#         biases: bool = False,
#         path_normalization: Optional[Union[str, float]] = None,
#         gradient_normalization: Optional[Union[str, float]] = None,
#         num_indexed_weights: Optional[int] = None,
#         weights_per_channel: bool = False,
#         force_irreps_out: bool = False,
#         weights_dim: Optional[int] = None,
#         input_dtype: torch.dtype = torch.get_default_dtype(),
#         linear_type: str = "vanilla",
#     ):
#         super(Linear, self).__init__()
#         irreps_in_regrouped = o3.Irreps(irreps_in).regroup()
#         irreps_out = o3.Irreps(irreps_out)

#         self.irreps_in = irreps_in_regrouped
#         self.channel_in = channel_in
#         self.channel_out = channel_out
#         self.biases = biases
#         self.path_normalization = path_normalization
#         self.num_indexed_weights = num_indexed_weights
#         self.weights_per_channel = weights_per_channel
#         self.force_irreps_out = force_irreps_out
#         self.linear_type = linear_type
#         self.weights_dim = weights_dim
#         self._input_dtype = input_dtype

#         self.gradient_normalization = _get_gradient_normalization(gradient_normalization)

#         channel_irrep_multiplier = 1
#         if self.channel_out is not None:
#             assert not self.weights_per_channel
#             channel_irrep_multiplier = self.channel_out

#         if not self.force_irreps_out:
#             irreps_out = irreps_out.filter(keep=irreps_in_regrouped)
#             irreps_out = irreps_out.simplify()
#         # This should factor in mul_to_axis somewhere
#         self.irreps_out = irreps_out

#         self._linear = FunctionalLinear(
#             irreps_in_regrouped,
#             channel_irrep_multiplier * irreps_out,
#             channel_out=self.channel_out,
#             channel_in=self.channel_in,
#             biases=self.biases,
#             path_normalization=self.path_normalization,
#             gradient_normalization=self.gradient_normalization,
#         )
#         self.device = device
#         self.weight_numel = self._linear.num_weights
#         self._weights = self._get_weights()

# def _get_weights(self):
#     """Constructs the weights for the linear module."""
#     irreps_in = self._linear.irreps_in
#     irreps_out = self._linear.irreps_out

#     weights = {}
#     instructions = []
#     for ins in self._linear.instructions:

#         if ins.i_in == -1:
#             name = f"b[{ins.i_out}]"
#         else:
#             name = f"w[{ins.i_in},{ins.i_out}] {irreps_in[ins.i_in]},{irreps_out[ins.i_out]}"

#         weight_shape = ins.path_shape
#         weight_std = ins.weight_std

#         weight = torch.nn.Parameter(
#             weight_std
#             * torch.randn(
#                 *weight_shape,
#                 dtype=self._input_dtype,
#             )
#         )
#         weights[name] = weight
#         instructions.append(
#             Instruction(
#                 i_in=ins.i_in,
#                 slice_in=ins.slice_in,
#                 i_out=ins.i_out,
#                 slice_out=ins.slice_out,
#                 path_shape=ins.path_shape,
#                 path_weight=ins.path_weight,
#                 weight_std=ins.weight_std,
#                 weight=weight.to(device=self.device),  # TODO (mit): This should not be happening
#             )
#         )
#     self._linear.instructions = instructions
#     return weights

# def forward(self, input) -> torch.Tensor:
#     """Apply the linear operator.

#     Args:
#         weights (optional IrrepsArray or jax.Array): scalar weights that are contracted with free parameters.
#             An array of shape ``(..., contracted_axis)``. Broadcasting with `input` is supported.
#         input (IrrepsArray): input irreps-array of shape ``(..., [channel_in,] irreps_in.dim)``.
#             Broadcasting with `weights` is supported.

#     Returns:
#         IrrepsArray: output irreps-array of shape ``(..., [channel_out,] irreps_out.dim)``.
#             Properly normalized assuming that the weights and input are properly normalized.
#     """
#     # Currently only supporting e3nn_jax.linear_vanilla
#     return self._linear(list(self._weights.values()), input)
