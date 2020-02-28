# pylint: disable=C,E1101,E1102
import unittest
from functools import partial

import torch

from e3nn.kernel import Kernel
from e3nn.radial import ConstantRadialModel
from e3nn.point.operations import Convolution
from e3nn.point.kernelconv import KernelConv
from e3nn.rs import dim


class TestKernelConv(unittest.TestCase):
    def setUp(self):
        super(TestKernelConv, self).setUp()
        torch.set_default_dtype(torch.float64)
        self.Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
        self.Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        batch = 100
        atoms = 40

        self.geometry = torch.rand(batch, atoms, 3)
        rb = self.geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = self.geometry.unsqueeze(2)  # [batch, a, 1, xyz]
        self.r = rb - ra
        self.features = torch.rand(batch, atoms, dim(self.Rs_in), requires_grad=True)
        self.mask = torch.ones(batch, atoms)
        torch.backends.cudnn.deterministic = True

        self.msg = "Convolution parameters were not identical. This means the test cannot compare outputs."

    def test_compare_forward_norm(self):
        for normalization in ["norm", "component"]:
            torch.manual_seed(0)
            K = partial(Kernel, RadialModel=ConstantRadialModel, normalization=normalization)
            C = Convolution(K, self.Rs_in, self.Rs_out)
            new_features = C(self.features, self.geometry) * self.mask.unsqueeze(dim=-1)

            torch.manual_seed(0)
            KC = KernelConv(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = KC(self.features, self.r, self.mask)

            assert all(torch.all(a == b) for a, b in zip(C.kernel.parameters(), KC.parameters())), self.msg
            self.assertTrue(torch.allclose(new_features, check_new_features))

    def test_compare_backward_features(self):
        check_features = self.features.clone().detach().requires_grad_()
        check_r = self.r.clone().detach()
        check_mask = self.mask.clone().detach()

        for normalization in ["norm", "component"]:
            torch.manual_seed(0)
            K = partial(Kernel, RadialModel=ConstantRadialModel, normalization=normalization)
            C = Convolution(K, self.Rs_in, self.Rs_out)
            new_features = C(self.features, self.geometry) * self.mask.unsqueeze(dim=-1)

            torch.manual_seed(0)
            KC = KernelConv(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = KC(check_features, check_r, check_mask)

            assert all(torch.all(a == b) for a, b in zip(C.kernel.parameters(), KC.parameters())), self.msg

            # Capture ground truth gradient
            target = torch.rand_like(new_features)
            loss = torch.norm(new_features - target)
            loss.backward()

            # Capture KernelConv gradient
            check_target = target.clone().detach()
            check_loss = torch.norm(check_new_features - check_target)
            check_loss.backward()

            self.assertTrue(torch.allclose(self.features.grad, check_features.grad))


if __name__ == '__main__':
    unittest.main()
