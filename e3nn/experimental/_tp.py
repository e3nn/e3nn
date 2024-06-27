from math import sqrt
from typing import List, Optional, Union, Any, Callable
import warnings

import torch
from torch import nn

import e3nn
from e3nn import o3
from e3nn.util import prod
from e3nn.o3._tensor_product._instruction import Instruction

class TensorProduct(nn.Module):
    def __init__(self,
                irreps_in1: o3.Irreps,
                irreps_in2: o3.Irreps,
                irreps_out: o3.Irreps,,
                instructions: List[tuple],
                in1_var: Optional[Union[List[float], torch.Tensor]] = None,,
                in2_var: Optional[Union[List[float], torch.Tensor]] = None,,
                out_var: Optional[Union[List[float], torch.Tensor]] = None,,
                irrep_normalization: str = None,
                path_normalization: str = None,
                internal_weights: Optional[bool] = None,
                shared_weights: Optional[bool] = None,
                ) -> None:
        
        super(TensorProduct, self).__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]
        
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out
        
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,),
                    "u<vw": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2, self.irreps_out[i_out].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]
        
        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.irreps_in1), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.irreps_in2), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.irreps_out), "Len of out_var must be equal to len(irreps_out)"
        
        def num_elements(ins):
            return {
                "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
            }[ins.connection_mode]
        
        normalization_coefficients = []

        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i) for i in instructions if i.i_out == ins.i_out)
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights        
        

        # === Determine weights ===
        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)

        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.Tensor())

        if self.irreps_out.dim > 0:
            output_mask = torch.cat(
                [
                    torch.ones(mul * ir.dim)
                    if any(
                        (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
                        for ins in self.instructions
                    )
                    else torch.zeros(mul * ir.dim)
                    for i_out, (mul, ir) in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = torch.ones(0)
        self.register_buffer("output_mask", output_mask)
        
        def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
            if weight is None:
                if self.weight_numel > 0 and not self.internal_weights:
                    raise RuntimeError("Weights must be provided when the TensorProduct does not have `internal_weights`")
                return self.weight
            else:
                if self.shared_weights:
                    torch._assert(weight.shape == (self.weight_numel,), "Invalid weight shape")
                else:
                    torch._assert(weight.shape[-1] == self.weight_numel, "Invalid weight shape")
                    torch._assert(weight.ndim > 1, "When shared weights is false, weights must have batch dimension")
            return weight
        
        def forward(self, x, y, weight: Optional[torch.Tensor] = None):

            torch._assert(x.shape[-1] == self._in1_dim, "Incorrect last dimension for x")
            torch._assert(y.shape[-1] == self._in2_dim, "Incorrect last dimension for y")

            real_weight = self._get_weights(weight)
            
            if shared_weights:
                output_shape = torch.broadcast_tensors(torch.empty((),).expand(x.shape[:-1]), torch.empty((),).expand(y.shape[:-1]))[0].shape
            else:
                output_shape = torch.broadcast_tensors(
                    torch.empty((),).expand(x.shape[:-1]), torch.empty((),).expand(y.shape[:-1]), torch.empty((),).expand(weight.shape[:-1])
                )[0].shape
                
            if len(self.instructions) == 0:
                output = x.new_zeros(output_shape + (irreps_out.dim,))
                return output
            
            
