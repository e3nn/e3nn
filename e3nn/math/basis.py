import math
import torch


def soft_one_hot_linspace(x, start, end, number, base='gaussian'):
    r"""Projection on a basis of gaussians

    .. math::

        y_i = \frac{1}{Z}\exp(-\left(\frac{x - r_i}{s}\right)^2)

    where :math:`x` is the input, :math:`r_i = R_0 + \frac{i}{N-1} (R_1 - R_0)` for :math:`i=0\dots N-1` and :math:`s=\frac{R_1-R_0}{N-1}`.
    :math:`Z` is set such that,

    .. math::

        \sum_i y_i^2 \approx 1

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis :math:`R_0`

    end : float
        maximum value span by the basis :math:`R_1`

    number : int
        number of gaussian functions :math:`N`

    base : {'gaussian', 'cosine'}

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
        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'gaussian'));

    .. jupyter-execute::

        x = torch.linspace(-1.0, 2.0, 100)
        plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, 3, 'cosine'));
    """
    if base == 'gaussian':
        sigma = (end - start) / (number - 1)
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)

        diff = x[..., None] - values  # [..., i]
        return diff.div(sigma).pow(2).neg().exp().div(1.12)

    if base == 'cosine':
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

        diff = x[..., None] - values
        return torch.cos(math.pi/2 * diff / step) * (diff < step) * (-step < diff)
