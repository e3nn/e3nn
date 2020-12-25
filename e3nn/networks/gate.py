# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name, abstract-method
from e3nn import rs
from e3nn.non_linearities import GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh


def make_gated_block(Rs_in, mul=16, lmax=3):
    scalars = [(mul, l, p) for mul, l, p in [(mul, 0, +1), (mul, 0, -1)] if rs.haslinearpath(Rs_in, l, p)]
    act_scalars = [(mul, swish if p == 1 else tanh) for mul, l, p in scalars]

    nonscalars = [(mul, l, p) for l in range(1, lmax + 1) for p in [+1, -1] if rs.haslinearpath(Rs_in, l, p)]
    if rs.haslinearpath(Rs_in, 0, +1):
        gates = [(rs.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, sigmoid)]
    else:
        gates = [(rs.mul_dim(nonscalars), 0, -1)]
        act_gates = [(-1, tanh)]

    return GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
