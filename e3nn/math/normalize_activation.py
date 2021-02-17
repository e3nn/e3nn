import torch

from e3nn.util import torch_default_tensor_type


def moment(f, n, dtype=torch.float64, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """
    with torch_default_tensor_type(dtype, device):
        # torch.Generator does not follow the default tensor's device type
        gen = torch.Generator(device=device).manual_seed(0)
        z = torch.randn(1_000_000, generator=gen)
        return f(z).pow(n).mean().item()


def normalize2mom(f, dtype=torch.float64, device=None):
    with torch_default_tensor_type(dtype, device):
        cst = 1 / moment(f, 2, dtype=dtype, device=device) ** 0.5
        if abs(cst - 1) < 1e-4:
            return f

        def g(z):
            return f(z).mul(cst)
        g.cst = cst
        return g
