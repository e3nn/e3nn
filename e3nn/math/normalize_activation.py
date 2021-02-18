import torch

from e3nn.util import torch_get_default_device, add_type_kwargs


@add_type_kwargs()
def moment(f, n):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """

    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64).to(
        dtype=torch.get_default_dtype(), device=torch_get_default_device())
    return f(z).pow(n).mean().item()


@add_type_kwargs()
def normalize2mom(f):
    cst = 1 / moment(f, 2) ** 0.5
    if abs(cst - 1) < 1e-4:
        return f

    def g(z):
        return f(z).mul(cst)
    g.cst = cst
    return g
