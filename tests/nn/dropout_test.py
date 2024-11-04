import copy

import torch

from e3nn.nn import Dropout
from e3nn.util.test import assert_auto_jitable, assert_equivariant
from e3nn.util.jit import prepare


def test_dropout() -> None:
    def build_module():
        return Dropout(irreps="10x1e + 10x0e", p=0.75)

    c = build_module()
    c_pt2 = torch.compile(prepare(build_module)(), fullgraph=True)
    x = c.irreps.randn(5, 2, -1)

    for c in [c, c_pt2, assert_auto_jitable(c)]:
        c.eval()
        assert c(x).eq(x).all()

        c.train()
        y = c(x)
        assert (y.eq(x / 0.25) | y.eq(0)).all()

        def wrap(x):
            torch.manual_seed(0)
            return c(x)

        assert_equivariant(wrap, args_in=[x], irreps_in=[c.irreps], irreps_out=[c.irreps])


def test_copy() -> None:
    c = Dropout(irreps="0e + 1e", p=0.5)
    _ = copy.deepcopy(c)
