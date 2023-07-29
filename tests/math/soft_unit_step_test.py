import torch

from e3nn.math import soft_unit_step


def test_grad() -> None:
    torch.set_default_dtype(torch.float64)
    x = torch.linspace(-1, 1, 1000, requires_grad=True)

    def f(x):
        return soft_unit_step(x).sum()

    assert torch.autograd.gradcheck(f, (x,), check_undefined_grad=False)


def test_grads() -> None:
    x = torch.linspace(-1, 1, 1000, requires_grad=True)

    y0 = soft_unit_step(x)
    assert torch.isfinite(y0).all()

    (y1,) = torch.autograd.grad(y0.sum(), x, create_graph=True)
    assert torch.isfinite(y1).all()

    (y2,) = torch.autograd.grad(y1.sum(), x, create_graph=True)
    assert torch.isfinite(y2).all()

    (y3,) = torch.autograd.grad(y2.sum(), x, create_graph=True)
    assert torch.isfinite(y3).all()
