import torch

from e3nn.o3 import Irreps
from e3nn.nn import Gate, Activation
from e3nn.nn.gate import _Sortcut
from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_activation():
    a = Activation("256x0o", [torch.abs])
    assert_auto_jitable(a)


def test_gate():
    irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_nonscalars = Irreps("16x0o"), [torch.tanh], Irreps("32x0o"), [torch.tanh], Irreps("16x1e+16x1o")

    sc = _Sortcut(irreps_scalars, irreps_gates)
    assert_auto_jitable(sc)

    g = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_nonscalars)
    assert_equivariant(g)
    assert_auto_jitable(g)
