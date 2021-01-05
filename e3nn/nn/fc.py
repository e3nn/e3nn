import torch

from e3nn.math import normalize2mom


def _identity(x):
    return x


class FullyConnectedNet(torch.nn.Module):
    r"""Fully-connected Neural Network

    Parameters
    ----------
    hs : list of int
        input, internal and output dimensions

    act : function
        activation funtion :math:`\phi`, it will be automatically normalized by a scaling factor such that

        .. math::

            \int_{-\infty}^{\infty} \phi(z)^2 \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz = 1
    """
    def __init__(self, hs, act=None, variance_in=1, variance_out=1, out_act=False):
        super().__init__()
        self.hs = tuple(hs)
        weights = []

        for h1, h2 in zip(self.hs, self.hs[1:]):
            weights.append(torch.nn.Parameter(torch.randn(h1, h2)))

        self.weights = torch.nn.ParameterList(weights)

        if act is None:
            act = _identity
        self.act = normalize2mom(act)
        self.variance_in = variance_in
        self.variance_out = variance_out
        self.out_act = out_act

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"

    def forward(self, x):
        """evaluate the network

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, hs[0])``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, hs[-1])``
        """
        with torch.autograd.profiler.record_function(repr(self)):
            for i, W in enumerate(self.weights):
                h_in, _h_out = W.shape

                if i == 0:  # first layer
                    W = W / (h_in * self.variance_in)**0.5
                if i > 0:  # not first layer
                    W = W / h_in**0.5
                if i == len(self.weights) - 1 and not self.out_act:  # last layer
                    W = W * self.variance_out**0.5

                x = x @ W

                if i < len(self.weights) - 1:  # not last layer
                    x = self.act(x)
                if i == len(self.weights) - 1 and self.out_act:  # last layer
                    x = self.act(x)

                if i == len(self.weights) - 1 and self.out_act:  # last layer
                    x = x * self.variance_out**0.5

            return x
