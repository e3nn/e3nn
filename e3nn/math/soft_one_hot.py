import math
import torch

from .soft_heaviside import soft_unit_step


def soft_one_hot_linspace(x, start, end, number, basis='gaussian', endpoint=True):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i\}_{i=1}^N`,

    .. math::

        y_i = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is set such that,

    .. math::

        \sum_{i=1}^N y_i^2 \approx 1

    See the last plot below.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'fourier', 'bessel', 'smooth_finite'}
        choice of basis family

    endpoint : bool
        if ``endpoint=False`` then for all :math:`x` outside of the interval defined by ``(start, end)``, :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = torch.linspace(-1.0, 2.0, 100)
        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'gaussian', endpoint=True));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'smooth_finite', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'cosine', endpoint=True));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'cosine', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'fourier', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'fourier', endpoint=True));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'bessel', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'bessel', endpoint=True));

    .. jupyter-execute::

        for basis in ['gaussian', 'cosine', 'fourier', 'smooth_finite']:
            for endpoint in [False, True]:
                y = soft_one_hot_linspace(x, -0.5, 1.5, 4, basis, endpoint)
                plt.plot(x, y.pow(2).sum(1), label=f"{basis} {'endpoint' if endpoint else ''}")
        plt.legend();
    """
    if endpoint:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    if basis == 'gaussian':
        return diff.pow(2).neg().exp().div(1.12)

    if basis == 'cosine':
        return torch.cos(math.pi/2 * diff) * (diff < 1) * (-1 < diff)

    if basis == 'smooth_finite':
        return 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)

    if basis == 'fourier':
        x = (x[..., None] - start) / (end - start)
        if endpoint:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == 'bessel':
        x = x[..., None] - start
        c = (end - start)
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out =  math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if endpoint:
            return out
        else:
            return out * ((x / c) < 1) * (0 < x)
