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
        biases: Optional[Union[List[bool], bool]] = False,
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

        irreps_in = o3.Irreps(irreps_in)
        irreps_in_slices = irreps_in.slices()
        irreps_out = o3.Irreps(irreps_out)
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

        def alpha(ins):
            x = sum(
                irreps_in[i.i_in if path_normalization == "element" else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if channel_in is not None:
                x *= channel_in
            return 1.0 if x == 0 else x

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
                torch.randn(*((channel_in, channel_out) if channel_in is not None else ()), self.weight_numel)
            )
        else:
            self.register_buffer("weight", torch.Tensor())

        # == Generate biases ==
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = torch.nn.Parameter(
                torch.zeros(*((channel_out,) if channel_out is not None else ()), self.bias_numel)
            )  # see appendix C.1 and Eq.5 of https://arxiv.org/pdf/2011.14522.pdf
        else:
            self.register_buffer("bias", torch.Tensor())

        # == Compute output mask ==
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
        self.irreps_out_count = [i for i, _ in enumerate(self.irreps_out)]
        self._irreps_out_dim = self.irreps_out.dim
        self._irreps_in_dim = self.irreps_in.dim
        self._irreps_in_len = len(self.irreps_in)
        self.instructions = instructions
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.register_buffer("output_mask", output_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(self, input, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            outsize = size + (self._irreps_out_dim,)
        else:
            size = input.shape[:-2]
            outsize = size + (
                self.channel_out,
                self._irreps_out_dim,
            )

        if self.bias_numel > 0:
            if self.channel_out is None:
                bias = bias.reshape(-1, self.bias_numel)
            else:
                bias = bias.reshape(-1, self.channel_out, self.bias_numel)

        if len(self.instructions) == 0 and self.bias_numel == 0:
            return input.new_zeros(outsize)

        if self.channel_in is None:
            input = input.reshape(-1, self._irreps_in_dim)
        else:
            input = input.reshape(-1, self.channel_in, self._irreps_in_dim)

        batch_out = input.shape[0]
        if self.weight_numel > 0:
            weight = (
                weight.reshape(-1, self.weight_numel)
                if self.channel_in is None
                else weight.reshape(-1, self.channel_in, self.channel_out, self.weight_numel)
            )

        # = extract individual input irreps =
        if self._irreps_in_len == 1:
            input_list = [
                input.reshape(
                    batch_out,
                    *(() if self.channel_in is None else (self.channel_in,)),
                    self.instructions[0].chunk_in.mul,
                    self.instructions[0].chunk_in.dim,
                )
            ]
        else:
            input_list = [
                input.narrow(-1, ins.chunk_in.slice.start, ins.chunk_in.mul * ins.chunk_in.dim).reshape(
                    batch_out, *(() if self.channel_in is None else (self.channel_in,)), ins.chunk_in.mul, ins.chunk_in.dim
                )
                for ins in self.instructions
            ]

        z = "" if self.shared_weights else "z"

        flat_weight_index = 0
        flat_bias_index = 0
        out_list = []

        for ins in self.instructions:
            mul_ir_out = ins.chunk_out.mul

            if ins.i_in == -1:
                # = bias =
                b = bias.narrow(-1, flat_bias_index, prod(ins.path_shape))
                flat_bias_index += prod(ins.path_shape)
                out_list += [
                    (ins.path_weight * b).reshape(
                        1, *(() if self.channel_out is None else (self.channel_out,)), mul_ir_out.dim
                    )
                ]
            else:
                mul_ir_in = ins.chunk_in.mul

                # Short-circut for empty irreps
                if ins.chunk_in.dim == 0 or ins.chunk_out.dim == 0:
                    continue

                # Extract the weight from the flattened weight tensor
                path_nweight = prod(ins.path_shape)
                if len(self.instructions) == 1:
                    # Avoid unnecessary view when there is only one weight
                    w = weight
                else:
                    w = weight.narrow(-1, flat_weight_index, path_nweight)
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
                    ein_out.reshape(
                        batch_out,
                        *(() if self.channel_out is None else (self.channel_out,)),
                        ins.chunk_out.mul * ins.chunk_out.dim,
                    )
                ]

        # = Return the result =
        out = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(
                    batch_out,
                    *(() if self.channel_out is None else (self.channel_out,)),
                    ins.chunk_out.mul * ins.chunk_out.dim,
                ),
                like=input,
            )
            for i_out in self.irreps_out_count
            if ins.chunk_out.mul > 0
        ]
        if len(out) > 1:
            out = torch.cat(out, dim=-1)
        else:
            out = out[0]

        out = out.reshape(outsize)
        return out
