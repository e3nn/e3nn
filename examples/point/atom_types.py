# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
from functools import partial

import torch

from e3nn import rs
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid, swish
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel


# Mixer is a wrapper around the Convolution operation
# It creates an operation for:
#        Many types of atoms --> One single type of atom
class Mixer(torch.nn.Module):
    def __init__(self, Op, Rs_in_s, Rs_out):
        super().__init__()
        self.ops = torch.nn.ModuleList([
            Op(Rs_in, Rs_out)
            for Rs_in in Rs_in_s
        ])

    def forward(self, *args, n_norm=1):
        # It simply sums the different outputs
        y = 0
        for m, x in zip(self.ops, args):
            y += m(*x, n_norm=n_norm)
        return y


R = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=3, act=swish)
K = partial(Kernel, RadialModel=R)
C = partial(Convolution, K)

Rs_in1 = [(2, 0), (1, 1)]
Rs_in2 = [(1, 1)]

Rs_out1 = [(3, 0), (1, 1)]
Rs_out2 = [(3, 1)]

act1 = GatedBlock(Rs_out1, swish, sigmoid)  # (Rs1, Rs2) --> Rs1
mix1 = Mixer(C, [Rs_in1, Rs_in2], act1.Rs_in)
act2 = GatedBlock(Rs_out2, swish, sigmoid)  # (Rs1, Rs2) --> Rs2
mix2 = Mixer(C, [Rs_in1, Rs_in2], act2.Rs_in)

# First type of atom (eg. Hydrogen)
# There is 2 atoms with their features and positions
fea1 = torch.randn(10, 2, rs.dim(Rs_in1))
geo1 = torch.randn(10, 2, 3)

# Second type of atom (eg. Oxygen)
# There is 3 atoms with their features and positions
fea2 = torch.randn(10, 3, rs.dim(Rs_in2))
geo2 = torch.randn(10, 3, 3)

# The layer is splited in two parts for the two output types of atoms
out1 = act1(mix1((fea1, geo1, geo1), (fea2, geo2, geo1), n_norm=5))  # output the features of the Hydrogen's
out2 = act2(mix2((fea1, geo1, geo2), (fea2, geo2, geo2), n_norm=5))  # output the features of the Oxygen's
