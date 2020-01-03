# pylint: disable=C,E1101,E1102
import unittest
import torch
from e3nn.image.gated_block import GatedBlock

class Tests(unittest.TestCase):
    def test_grad(self):
        m = GatedBlock([1, 1, 1], [1, 1, 1], 5, activation=(torch.relu, torch.sigmoid), checkpoint=False).type(torch.float64)

        x = torch.rand(1, 1 + 3 + 5, 6, 6, 6, requires_grad=True, dtype=torch.float64)

        self.assertTrue(torch.autograd.gradcheck(m, (x, ), eps=1e-3))


unittest.main()
