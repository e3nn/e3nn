from functools import partial

import torch

from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid, swish
from e3nn.point.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.point.radial import CosineBasisModel


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
M = partial(Mixer, C)  # wrap C to accept many input types

Rs_in1 = [(2, 0), (1, 1)]
Rs_in2 = [(1, 1)]

Rs_out1 = [(3, 0), (1, 1)]
Rs_out2 = [(3, 1)]

m1 = GatedBlock(partial(M, [Rs_in1, Rs_in2]), Rs_out1, swish, sigmoid)  # (Rs1, Rs2) --> Rs1
m2 = GatedBlock(partial(M, [Rs_in1, Rs_in2]), Rs_out2, swish, sigmoid)  # (Rs1, Rs2) --> Rs2
# partial(M, (Rs1, Rs2)) is an operation of signature (Rs_out), it will be instantiate by GatedBlock

# First type of atom (eg. Hydrogen)
# There is 2 atoms with their features and positions
fea1 = torch.randn(10, 2, sum(mul * (2 * l + 1) for mul, l in Rs_in1))
geo1 = torch.randn(10, 2, 3)

# Second type of atom (eg. Oxygen)
# There is 3 atoms with their features and positions
fea2 = torch.randn(10, 3, sum(mul * (2 * l + 1) for mul, l in Rs_in2))
geo2 = torch.randn(10, 3, 3)

# The layer is splited in two parts for the two output types of atoms
out1 = m1((fea1, geo1, geo1), (fea2, geo2, geo1), n_norm=5)  # output the features of the Hydrogen's
out2 = m2((fea1, geo1, geo2), (fea2, geo2, geo2), n_norm=5)  # output the features of the Oxygen's
