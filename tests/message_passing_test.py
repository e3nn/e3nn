# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest
from functools import partial

import torch
from e3nn import rs, o3
from e3nn.point.message_passing import E3Conv

from e3nn.radial import ConstantRadialModel
from e3nn.kernel import Kernel


class Tests(unittest.TestCase):

    def test_E3conv(self):
        torch.set_default_dtype(torch.float64)

        lmax = 2
        Rs_in = [(1, 2)]
        Rs_out = [(1, 3)]

        RadialModel = ConstantRadialModel

        K = partial(Kernel, RadialModel=RadialModel,
                    selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

        conv = E3Conv(K, Rs_in, Rs_out)

        N = 9
        c_in = rs.dim(Rs_in)
        c_out = rs.dim(Rs_out)
        x = torch.arange(0, c_in*N, dtype=torch.float64).reshape(N, c_in)
        edge_index = torch.LongTensor(
            [[0, 0, 1, 1, 3, 4],
             [1, 2, 0, 2, 6, 5]]
        )
        # edge_attr is the radii of the relative distance vectors
        edge_attr = torch.randn(edge_index.shape[-1], 3)
        out = conv(x, edge_index, edge_attr, size=(N, N))
        assert list(out.shape) == [N, c_out]


if __name__ == '__main__':
    unittest.main()
