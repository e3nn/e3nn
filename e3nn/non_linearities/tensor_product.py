# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch
from e3nn.non_linearities.multiplication import Multiplication
from e3nn.non_linearities.self_interaction import SelfInteraction


class TensorProduct(torch.nn.Module):
    def __init__(self, Operation, Rs_out, Rs_mid, mul_mid):
        super().__init__()
        self.mul_mid = mul_mid
        self.f1 = Operation(mul_mid * Rs_mid)
        self.f2 = Operation(mul_mid * Rs_mid)
        self.m = Multiplication(Rs_mid, Rs_mid)
        self.si = SelfInteraction(mul_mid * self.m.Rs_out, Rs_out)

    def forward(self, *args):
        """
        :return:         tensor [..., channel]
        """
        x1 = self.f1(*args)
        x2 = self.f2(*args)

        # split into mul x Rs
        *size, _ = x1.shape
        x1 = x1.view(*size, self.mul_mid, -1)
        x2 = x2.view(*size, self.mul_mid, -1)

        x = self.m(x1, x2).view(*size, -1)
        x = self.si(x)
        return x
