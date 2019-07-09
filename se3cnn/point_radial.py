# pylint: disable=C, R, arguments-differ, no-member
import math
from functools import partial

import torch


class FiniteElementModel(torch.nn.Module):
    def __init__(self, position, basis, Model, out_dim):
        '''
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        :param out_dim: output dimension
        '''
        super().__init__()
        self.register_buffer('position', position)
        self.basis = basis
        self.f = Model(len(position), out_dim)

    def forward(self, x):
        """
        :param x: tensor [batch, ...]
        :return: tensor [batch, dim]
        """
        diff = x.unsqueeze(1) - self.position.unsqueeze(0)  # [batch, i, ...]
        batch, n, *rest = diff.size()
        x = self.basis(diff.view(-1, *rest)).view(batch, n)  # [batch, i]
        return self.f(x)


class FC(torch.nn.Module):
    def __init__(self, d1, d2, h, L, act):
        super().__init__()

        weights = []

        hh = d1
        for _ in range(L):
            weights.append(torch.nn.Parameter(torch.randn(h, hh)))
            hh = h

        weights.append(torch.nn.Parameter(torch.randn(d2, hh)))
        self.weights = torch.nn.ParameterList(weights)
        self.act = act

    def forward(self, x):
        L = len(self.weights) - 1

        if L == 0:
            W = self.weights[0]
            h = x.size(1)
            return x @ (W.t() / h ** 0.5)

        for i, W in enumerate(self.weights):
            h = x.size(1)

            if i == 0:
                x = self.act(x @ W.t())
            elif i < L:
                x = self.act(x @ (W.t() / h ** 0.5))
            else:
                x = x @ (W.t() / h)

        return x


def GaussianBasisModel(max_radius, number_of_basis, number_of_hidden_layers, number_of_hidden_neurons, out_dim):
    radii = torch.linspace(0, max_radius, steps=number_of_basis)
    sigma = 0.45 * (radii[1] - radii[0])
    basis = lambda x: x.pow(2).div(2 * sigma ** 2).neg().exp()
    Model = partial(FC, h=number_of_hidden_neurons, L=number_of_hidden_layers, act=lambda x: x.relu().mul(2 ** 0.5))
    return FiniteElementModel(radii, basis, Model, out_dim)


def CosineBasisModel(max_radius, number_of_basis, number_of_hidden_layers, number_of_hidden_neurons, out_dim):
    radii = torch.linspace(0, max_radius, steps=number_of_basis)
    step = radii[1] - radii[0]
    basis = lambda x: x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)
    Model = partial(FC, h=number_of_hidden_neurons, L=number_of_hidden_layers, act=lambda x: x.relu().mul(2 ** 0.5))
    return FiniteElementModel(radii, basis, Model, out_dim)
