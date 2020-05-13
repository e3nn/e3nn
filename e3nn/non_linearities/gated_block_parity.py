# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, unbalanced-tuple-unpacking
import torch

from e3nn import rs
from e3nn.non_linearities.activation import Activation


class GatedBlockParity(torch.nn.Module):
    def __init__(self, Rs_scalars, act_scalars, Rs_gates, act_gates, Rs_nonscalars):
        super().__init__()

        self.Rs_in = Rs_scalars + Rs_gates + Rs_nonscalars
        self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars = Rs_scalars, Rs_gates, Rs_nonscalars

        self.act_scalars = Activation(Rs_scalars, act_scalars)
        Rs_scalars = self.act_scalars.Rs_out

        self.act_gates = Activation(Rs_gates, act_gates)
        Rs_gates = self.act_gates.Rs_out

        self.mul = rs.ElementwiseTensorProduct(Rs_nonscalars, Rs_gates)
        Rs_nonscalars = self.mul.Rs_out

        self.Rs_out = Rs_scalars + Rs_nonscalars

    def __repr__(self):
        return "{name} ({Rs_scalars} + {Rs_gates} + {Rs_nonscalars} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_scalars=rs.format_Rs(self.Rs_scalars),
            Rs_gates=rs.format_Rs(self.Rs_gates),
            Rs_nonscalars=rs.format_Rs(self.Rs_nonscalars),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features, dim=-1):
        scalars, gates, nonscalars = rs.cut(features, self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars, dim_=dim)
        scalars = self.act_scalars(scalars)
        if gates.shape[dim]:
            gates = self.act_gates(gates)
            nonscalars = self.mul(nonscalars, gates)
            features = torch.cat([scalars, nonscalars], dim=dim)
        else:
            features = scalars
        return features
