# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import math
from functools import partial

import torch


class ConstantRadialModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d))

    def forward(self, _radii):
        batch = _radii.size(0)
        return self.weight.reshape(1, -1).expand(batch, -1)


class FiniteElementModel(torch.nn.Module):
    def __init__(self, out_dim, position, basis, Model):
        '''
        :param out_dim: output dimension
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
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
        x = self.basis(diff.reshape(-1, *rest)).reshape(batch, n)  # [batch, i]
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
                # note: normalization assumes that the sum of the inputs is 1
                x = self.act(x @ W.t())
            elif i < L:
                x = self.act(x @ (W.t() / h ** 0.5))
            else:
                x = x @ (W.t() / h ** 0.5)

        return x


def FiniteElementFCModel(out_dim, position, basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteElementModel(out_dim, position, basis, Model)


def CosineBasisModel(out_dim, max_radius, number_of_basis, h, L, act):
    r""" The tex for the math reads
    \begin{cases}
        cos^2 (\gamma (d_{ab} - \mu_{k}) \frac{\pi}{2}) & 1 \geq \gamma (d_{ab} - \mu_{k}) \geq -1 \\
        0 & otherwise
    \end{cases}
    """
    radii = torch.linspace(0, max_radius, steps=number_of_basis)
    step = radii[1] - radii[0]

    def basis(x):
        return x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)
    return FiniteElementFCModel(out_dim, radii, basis, h, L, act)


def GaussianRadialModel(out_dim, max_radius, number_of_basis, h, L, act, min_radius=0.):
    """exp(-x^2 /spacing)"""
    spacing = (max_radius - min_radius) / (number_of_basis - 1)
    radii = torch.linspace(min_radius, max_radius, number_of_basis)
    sigma = 0.8 * spacing

    def basis(x):
        return x.div(sigma).pow(2).neg().exp().div(1.423085244900308)
    return FiniteElementFCModel(out_dim, radii, basis, h, L, act)


class BesselRadialModel(torch.nn.Module):
    r"""The tex for the math reads
    \begin{cases}
        \sqrt{\frac{2}{c}} \frac{sin(\frac{n \pi}{c} d)}{d} & 0 \leq x \leq max_radius \\
        0 & otherwise
    \end{cases}
    c = max_radius (cutoff), n = number_of_basis, d = distance (in R_{+})"""
    def __init__(self, out_dim, max_radius, number_of_basis, h, L, act, epsilon=1e-8):
        super().__init__()
        n = torch.linspace(1, number_of_basis, number_of_basis).unsqueeze(0)
        n_scaled = n * math.pi / max_radius
        self.register_buffer('n_scaled', n_scaled)
        self.factor = math.sqrt(2 / max_radius)
        self.max_radius = max_radius
        self.epsilon = epsilon

        self.f = FC(number_of_basis, out_dim, h=h, L=L, act=act)

    def basis(self, x):
        assert x.ndim == 1
        x_within_cutoff = ((x >= 0.0) * (x <= self.max_radius)) * x
        x_within_cutoff = x_within_cutoff.unsqueeze(-1)
        return self.factor * torch.sin(self.n_scaled * x_within_cutoff) / (x_within_cutoff + self.epsilon)

    def forward(self, x):
        x = self.basis(x)
        return self.f(x)
