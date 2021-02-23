import math
import torch


def soft_one_hot_linspace(x, start, end, number, base='gaussian', endpoint=True):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i\}_{i=1}^N`,

    .. math::

        y_i = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith base function.
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

    base : {'gaussian', 'cosine', 'fourier'}
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

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'cosine', endpoint=True));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'cosine', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'fourier', endpoint=False));

    .. jupyter-execute::

        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'fourier', endpoint=True));

    .. jupyter-execute::

        for base in ['gaussian', 'cosine', 'fourier']:
            for endpoint in [False, True]:
                y = soft_one_hot_linspace(x, -0.5, 1.5, 4, base, endpoint)
                plt.plot(x, y.pow(2).sum(1), label=f"{base} {'endpoint' if endpoint else ''}")
        plt.legend();
    """
    if endpoint:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = x[..., None] - values

    if base == 'gaussian':
        return diff.div(step).pow(2).neg().exp().div(1.12)

    if base == 'cosine':
        return torch.cos(math.pi/2 * diff / step) * (diff < step) * (-step < diff)

    if base == 'fourier':
        x = (x[..., None] - start) / (end - start)
        if endpoint:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(number / 2) * (0 < x) * (x < 1)
