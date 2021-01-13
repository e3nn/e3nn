import torch


def soft_one_hot_linspace(x, start, end, number):
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

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`
    """
    sigma = (end - start) / (number - 1)
    values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)

    diff = x[..., None] - values  # [..., i]
    return diff.div(sigma).pow(2).neg().exp().div(1.12)
