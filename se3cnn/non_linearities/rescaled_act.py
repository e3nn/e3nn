# pylint: disable=no-member, missing-docstring, invalid-name
import torch


class Softplus:
    def __init__(self, beta):
        x = torch.randn(100000, dtype=torch.float64)
        self.factor = torch.nn.functional.softplus(x, beta).pow(2).mean().rsqrt().item()
        self.beta = beta

    def __call__(self, x):
        return torch.nn.functional.softplus(x, self.beta).mul(self.factor)


class ShiftedSoftplus:
    def __init__(self, beta):
        x = torch.randn(100000, dtype=torch.float64)
        self.factor = torch.nn.functional.softplus(x, beta).pow(2).mean().rsqrt().item()
        self.shift = torch.nn.functional.softplus(torch.zeros(()), beta).item()
        self.beta = beta

    def __call__(self, x):
        return torch.nn.functional.softplus(x, self.beta).sub(self.shift).mul(self.factor)


def sigmoid(x):
    return x.sigmoid().mul(1.84623)


def tanh(x):
    return x.tanh().mul(1.59254)


def relu(x):
    return x.relu().mul(2 ** 0.5)
