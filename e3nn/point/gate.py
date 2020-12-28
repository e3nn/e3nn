import torch

from e3nn import rs
from e3nn.rs import TY_RS_STRICT
from e3nn import tensor_block

from itertools import accumulate


class Gate(torch.nn.Module):
    def __init__(self, Rs_out: TY_RS_STRICT, scalar_act, tensor_act):
        super().__init__()
        self.Rs_out = Rs_out

        self.n_scalars = Rs_out[0][0] if Rs_out[0][1] == 0 else 0
        self.n_gates = sum(mul for (mul, l, _) in Rs_out if l > 0)
        self.n_tensor_entries = rs.dim(Rs_out) - self.n_scalars

        self.Rs_in = rs.simplify([(self.n_gates, 0, 0)] + Rs_out)
        self.Rs_in_truncated = [(mul, l, p) for (mul, l, p) in self.Rs_in if l != 0]

        self.register_buffer('L_list', torch.tensor(rs.extract_l(self.Rs_in_truncated), dtype=torch.int32))
        self.register_buffer('mul_sizes', torch.tensor(rs.extract_mul(self.Rs_in_truncated), dtype=torch.int32))

        self.register_buffer('tensor_offsets', torch.tensor([0] + list(accumulate(mul * (2*l + 1) for (mul, l) in zip(self.mul_sizes, self.L_list))), dtype=torch.int32))
        self.register_buffer('gate_offsets', torch.tensor([0] + list(accumulate(self.mul_sizes)), dtype=torch.int32))

        self.scalar_act = scalar_act
        self.tensor_act = tensor_act

    def forward(self, x):
        batch_size = x.size(0)
        representation_size = x.size(1) - self.n_gates
        output = x.new_empty((batch_size, representation_size))

        gates = x.narrow(dim=1, start=0, length=self.n_gates)
        scalars = x.narrow(dim=1, start=self.n_gates, length=self.n_scalars)
        tensors = x.narrow(dim=1, start=self.n_gates+self.n_scalars, length=self.n_tensor_entries)

        if self.n_scalars > 0:
            output[:, :self.n_scalars] = self.scalar_act(scalars)

        if self.n_tensor_entries > 0:
            output[:, self.n_scalars:] = tensor_gate(tensors, self.tensor_act(gates), self)

        return output


class GateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gate, gate_class_instance):
        ctx.save_for_backward(input, gate)
        ctx.gate_class_instance = gate_class_instance

        gate_repeated = tensor_block.repeat_m(
            gate,
            L_list=gate_class_instance.L_list,
            mul_sizes=gate_class_instance.mul_sizes,
            input_offsets=gate_class_instance.gate_offsets,
            output_offsets=gate_class_instance.tensor_offsets)

        return gate_repeated * input

    @staticmethod
    def backward(ctx, grad_outputs):
        input, gate = ctx.saved_tensors
        gate_class_instance = ctx.gate_class_instance

        gate_repeated = tensor_block.repeat_m(
            gate,
            L_list=gate_class_instance.L_list,
            mul_sizes=gate_class_instance.mul_sizes,
            input_offsets=gate_class_instance.gate_offsets,
            output_offsets=gate_class_instance.tensor_offsets)

        grad_input = gate_repeated * grad_outputs

        grad_gate = tensor_block.sum_m(
            input * grad_outputs,
            L_list=gate_class_instance.L_list,
            mul_sizes=gate_class_instance.mul_sizes,
            input_offsets=gate_class_instance.tensor_offsets,
            output_offsets=gate_class_instance.gate_offsets)

        return grad_input, grad_gate, None


tensor_gate = GateFunction.apply
