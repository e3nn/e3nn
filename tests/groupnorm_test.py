# pylint: disable=C,E1101,E1102
import unittest

import torch

from e3nn.groupnorm import GroupNorm


class Tests(unittest.TestCase):
    def test_groupnorm(self):
        bn = GroupNorm([(3, 1), (4, 3), (1, 5)])

        x = torch.rand(16, 3 + 12 + 5, 10, 10, 10)
        y = bn(x)
        return y


if __name__ == '__main__':
    unittest.main()
