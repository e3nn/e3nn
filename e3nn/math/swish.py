import torch


@torch.jit.script
def _swish_jit_fwd(x):
    return x * torch.sigmoid(x) * 1.679176792398942


@torch.jit.script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid))) * 1.679176792398942


class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        x = ctx.saved_tensors[0]
        return _swish_jit_bwd(x, grad_output)


def swish(x):
    r"""swish function

    .. math:: s(x) \propto x \sigma(x)

    where :math:`\sigma` is the sigmoid

    .. math:: \sigma(x) = e^x / (1 + e^x)

    :math:`s` is normalized such that

    .. math:: \int s(x)^2 d\mu(x) = 1

    where :math:`\mu` is the normal measure.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape ``(...)``

    Returns
    -------
    `torch.Tensor`
        tensor of shape ``(...)``

    Examples
    --------

    >>> x = torch.randn(100_000)
    >>> s = swish(x)
    >>> a = s.pow(2).mean()
    >>> a.log().abs().item() < 0.05  # 0.95 < a < 1.05
    True
    """
    return _SwishJitAutoFn.apply(x)
