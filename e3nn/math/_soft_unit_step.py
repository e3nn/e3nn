import torch


class _SoftUnitStep(torch.autograd.Function):
    # pylint: disable=arguments-differ

    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = torch.zeros_like(x)
        mask = x > 0.0
        safe_x = torch.where(mask, x, torch.ones_like(x))  # Avoid division by zero
        y = torch.where(mask, torch.exp(-1.0 / safe_x), torch.zeros_like(x))
        return y

    @staticmethod
    def backward(ctx, dy) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        mask = x > 0.0
        safe_x = torch.where(mask, x, torch.ones_like(x))  # Avoid division by zero
        dx = torch.where(mask, torch.exp(-1.0 / safe_x) / (safe_x * safe_x), torch.zeros_like(x))
        return dx * dy


def soft_unit_step(x):
    r"""smooth :math:`C^\infty` version of the unit step function

    .. math::

        x \mapsto \theta(x) e^{-1/x}


    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(...)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_unit_step
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = torch.linspace(-1.0, 10.0, 1000)
        plt.plot(x, soft_unit_step(x));
    """
    return _SoftUnitStep.apply(x)
