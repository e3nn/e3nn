# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name, abstract-method
import torch

from e3nn import rs
from e3nn.non_linearities.rescaled_act import swish, tanh
from e3nn.non_linearities.s2 import S2Activation
from e3nn.tensor_product import LearnableTensorSquare


class S2Network(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, layers=3):
        super().__init__()

        Rs = self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)
        self.act = S2Activation(list(range(lmax + 1)),
                                swish, res=20 * (lmax + 1))

        self.layers = []

        for _ in range(layers):
            lin = LearnableTensorSquare(
                Rs, mul * self.act.Rs_in, linear=True, allow_zero_outputs=True)

            # s2 nonlinearity
            Rs = mul * self.act.Rs_out

            self.layers += [lin]

        self.layers = torch.nn.ModuleList(self.layers)

        self.tail = LearnableTensorSquare(Rs, self.Rs_out)

    def forward(self, x):
        for lin in self.layers:
            x = lin(x)

            # put multiplicity into batch
            x = x.reshape(*x.shape[:-1], -1, rs.dim(self.act.Rs_in))
            x = self.act(x)
            x = x.reshape(*x.shape[:-2], -1)  # put back into representation

        x = self.tail(x)
        return x


class S2ParityNetwork(torch.nn.Module):
    def __init__(self, Rs_in, mul, lmax, Rs_out, layers=3):
        super().__init__()

        Rs = self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        def make_act(p_val, p_arg, act):
            Rs = [(1, l, p_val * p_arg**l) for l in range(lmax + 1)]
            return S2Activation(Rs, act, res=20 * (lmax + 1))

        self.act1, self.act2 = make_act(1, -1, swish), make_act(-1, -1, tanh)
        self.mul = mul

        self.layers = []

        for _ in range(layers):
            Rs_out = mul * (self.act1.Rs_in + self.act2.Rs_in)
            lin = LearnableTensorSquare(
                Rs, Rs_out, linear=True, allow_zero_outputs=True)

            # s2 nonlinearity
            Rs = mul * (self.act1.Rs_out + self.act2.Rs_out)

            self.layers += [lin]

        self.layers = torch.nn.ModuleList(self.layers)

        self.tail = LearnableTensorSquare(Rs, self.Rs_out)

    def forward(self, x):
        for lin in self.layers:
            x = lin(x)

            # put multiplicity into batch
            x = x.reshape(*x.shape[:-1], self.mul, -1)
            x1 = x.narrow(-1, 0, rs.dim(self.act1.Rs_in))
            x2 = x.narrow(-1, rs.dim(self.act1.Rs_in), rs.dim(self.act2.Rs_in))
            x1 = self.act1(x1)
            x2 = self.act2(x2)
            x = torch.cat([x1, x2], dim=-1)
            x = x.reshape(*x.shape[:-2], -1)  # put back into representation

        x = self.tail(x)
        return x
