import torch

from e3nn.util import explicit_default_types
from e3nn.util.jit import compile_mode


def moment(f, n, dtype=None, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """

    dtype, device = explicit_default_types(dtype, device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)
    return f(z).pow(n).mean()


@compile_mode('trace')
class normalize2mom(torch.nn.Module):
    def __init__(self, f, dtype=None, device=None):
        super().__init__()
        cst = 1 / moment(f, 2, dtype=dtype, device=device) ** 0.5

        if abs(cst - 1) < 1e-4:
            self._is_id = True
        else:
            self._is_id = False

        self.f = f
        self.register_buffer('cst', cst)

    def forward(self, x):
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x).mul(self.cst)

    def _make_tracing_inputs(self, n: int):
        # No reason to trace this with more than one tiny input,
        # since f is assumed by `moment` to be an elementwise scalar
        # function
        return [{
            'forward': (torch.zeros(
                size=(1,),
                dtype=self.cst.dtype,
                device=self.cst.device
            ),)
        }]
