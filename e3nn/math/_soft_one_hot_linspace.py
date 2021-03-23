import math
import torch

from e3nn.math import soft_unit_step


def soft_one_hot_linspace(x, start, end, number, basis=None, cutoff=None):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

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

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of other basis choices

    cutoff : bool
        if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``, :math:`\forall i, \; f_i(x) \approx 0`

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

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = torch.linspace(-1.0, 2.0, 100)

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified")

    if not cutoff:
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
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == 'bessel':
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f"basis=\"{basis}\" is not a valid entry")
