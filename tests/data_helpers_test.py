# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import unittest

import torch
import e3nn.data_helpers as dh
from e3nn import rs


class Tests(unittest.TestCase):
    def test_data_helpers(self):
        N = 7
        lattice = torch.randn(3, 3)
        pos = torch.randn(N, 3)
        Rs_in = [(3, 0), (1, 1)]
        x = torch.randn(N, rs.dim(Rs_in))
        r_max = 1
        indices, rel_vec = dh.neighbor_list_and_relative_vec_lattice(pos, lattice, r_max)
        data = dh.DataPeriodicNeighbors(x, Rs_in, pos, lattice, r_max)
        indices, rel_vec = dh.neighbor_list_and_relative_vec(pos, r_max)
        data = dh.DataNeighbors(x, Rs_in, pos, r_max)


if __name__ == '__main__':
    unittest.main()