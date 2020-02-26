# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name
import unittest
import torch
from e3nn.dropout import Dropout


class Tests(unittest.TestCase):
    def test_that_it_runs(self):
        Rs = [(3, 1), (4, 3)]
        m = Dropout(Rs)

        x = torch.randn(10, sum(mul * d for mul, d in Rs), dtype=torch.float64)
        m(x)

        x = torch.randn(10, sum(mul * d for mul, d in Rs), dtype=torch.float32)
        m(x)


if __name__ == '__main__':
    unittest.main()
