import torch

from e3nn.util import torch_default_tensor_type, torch_get_default_device


def moment(f, n, dtype=None, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """
    # must explicitly set to the default dtype and device, since Tensor.to(dtype|device=None) is no-op
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch_get_default_device()

    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)
    return f(z).pow(n).mean().item()


def normalize2mom(f, dtype=None, device=None):
    with torch_default_tensor_type(dtype, device):
        cst = 1 / moment(f, 2) ** 0.5
        if abs(cst - 1) < 1e-4:
            return f

        def g(z):
            return f(z).mul(cst)
        g.cst = cst
        return g
