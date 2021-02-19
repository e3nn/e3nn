import torch

from e3nn.util import explicit_default_types


def moment(f, n, dtype=None, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """

    dtype, device = explicit_default_types(dtype, device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)
    return f(z).pow(n).mean().item()


def normalize2mom(f, dtype=None, device=None):
    cst = 1 / moment(f, 2, dtype=dtype, device=device) ** 0.5
    if abs(cst - 1) < 1e-4:
        return f

    def g(z):
        return f(z).mul(cst)
    g.cst = cst
    return g
