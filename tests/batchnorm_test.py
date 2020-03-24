# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name
import unittest
import torch
from e3nn.batchnorm import BatchNorm


class Tests(unittest.TestCase):
    def test_normalization(self):
        torch.set_default_dtype(torch.float64)

        batch, n = 20, 20
        Rs = [(3, 1), (4, 3)]

        m = BatchNorm(Rs, normalization='norm')

        x = torch.randn(batch, n, sum(mul * d for mul, d in Rs)).mul(5.0).add(10.0)
        x = m(x)

        a = x[..., :3]  # [batch, space, mul]
        assert a.mean([0, 1]).abs().max() < 1e-10
        assert a.pow(2).mean([0, 1]).sub(1).abs().max() < 1e-5

        a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
        assert a.pow(2).sum(3).mean([0, 1]).sub(1).abs().max() < 1e-5

        #

        m = BatchNorm(Rs, normalization='component')

        x = torch.randn(batch, n, sum(mul * d for mul, d in Rs)).mul(5.0).add(10.0)
        x = m(x)

        a = x[..., :3]  # [batch, space, mul]
        assert a.mean([0, 1]).abs().max() < 1e-10
        assert a.pow(2).mean([0, 1]).sub(1).abs().max() < 1e-5

        a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
        assert a.pow(2).mean(3).mean([0, 1]).sub(1).abs().max() < 1e-5


if __name__ == '__main__':
    unittest.main()
