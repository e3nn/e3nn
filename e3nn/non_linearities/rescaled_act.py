# pylint: disable=no-member, missing-docstring, invalid-name, arguments-differ
"""
rescaled activation functions such that the second moment is equal to one

int Dx f(x)^2 = 1

where Dx is the gaussian measure
"""
import torch


class Softplus:
    def __init__(self, beta):
        x = torch.randn(100000, dtype=torch.float64)
        self.factor = torch.nn.functional.softplus(x, beta).pow(2).mean().rsqrt().item()
        self.beta = beta

    def __call__(self, x):
        return torch.nn.functional.softplus(x, self.beta).mul(self.factor)


softplus = Softplus(1)


class ShiftedSoftplus:
    def __init__(self, beta):
        x = torch.randn(100000, dtype=torch.float64)
        self.shift = torch.nn.functional.softplus(torch.zeros(()), beta).item()
        y = torch.nn.functional.softplus(x, beta).sub(self.shift)
        self.factor = y.pow(2).mean().rsqrt().item()
        self.beta = beta

    def __call__(self, x):
        return torch.nn.functional.softplus(x, self.beta).sub(self.shift).mul(self.factor)


shiftedsoftplus = ShiftedSoftplus(1)


def identity(x):
    return x


def quadratic(x):
    return x.pow(2)


def sigmoid(x):
    return x.sigmoid().mul(1.84623)


def tanh(x):
    return x.tanh().mul(1.59254)


def relu(x):
    return x.relu().mul(2 ** 0.5)


def absolute(x):
    return x.abs()


@torch.jit.script
def swish_jit_fwd(x):
    return x * torch.sigmoid(x) * 1.6768


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid))) * 1.6768


class SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x):
    return SwishJitAutoFn.apply(x)
