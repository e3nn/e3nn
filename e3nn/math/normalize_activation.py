import torch

from e3nn.util import torch_default_dtype

@torch_default_dtype(torch.float64)
def moment(f, n, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """
    gen = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, device=device)
    return f(z).pow(n).mean().item()


def normalize2mom(f, device=None):
    cst = 1 / moment(f, 2, device=device) ** 0.5
    if abs(cst - 1) < 1e-4:
        return f

    def g(z):
        return f(z).mul(cst)
    g.cst = cst
    return g
