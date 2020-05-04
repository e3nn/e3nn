# pylint: disable=C,E1101,E1102
import unittest

import torch

from e3nn.image.filter import low_pass_filter


class Tests(unittest.TestCase):
    def test_low_pass_filter(self):
        x = torch.randn(32, 32, 32)
        low_pass_filter(x, 2.0, 2)


if __name__ == '__main__':
    unittest.main()
