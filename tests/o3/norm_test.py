import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant, assert_auto_jitable, random_irreps
from e3nn.util.jit import get_optimization_defaults, set_optimization_defaults


@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4))
@pytest.mark.parametrize("squared", [True, False])
def test_norm(irreps_in, squared) -> None:

    m = o3.Norm(irreps_in, squared=squared)
    m(torch.randn(m.irreps_in.dim))
    if m.irreps_in.dim == 0:
        return
    assert_equivariant(m)
    assert_auto_jitable(m)

    # Turning off the torch.jit.script in CodeGenMix to enable torch.compile.
    jit_mode_before = get_optimization_defaults()["jit_mode"]
    try:
        set_optimization_defaults(jit_mode="inductor")
        m = o3.Norm(irreps_in, squared=squared)
        torch._dynamo.reset() # Clear cache from the previous run
        m_pt2 = torch.compile(m, fullgraph=True)
        m_pt2(torch.randn(m_pt2.irreps_in.dim))
    finally:
        set_optimization_defaults(jit_mode=jit_mode_before)



@pytest.mark.parametrize("squared", [True, False])
def test_grad(squared) -> None:
    """Confirm has zero grad at zero"""
    irreps_in = o3.Irreps("2x0e + 3x0o")
    norm = o3.Norm(irreps_in, squared=squared)
    with torch.autograd.set_detect_anomaly(True):
        inp = torch.zeros(norm.irreps_in.dim, requires_grad=True)
        out = norm(inp)
        grads = torch.autograd.grad(
            outputs=out.sum(),
            inputs=inp,
        )[0]
        assert torch.allclose(grads, torch.zeros(1))


@pytest.mark.parametrize("squared", [True, False])
def test_vector_norm(squared) -> None:
    n = 10
    batch = 3
    irreps_in = o3.Irreps([(n, (1, -1))])
    vecs = torch.randn(batch, n, 3)
    norm_mod = o3.Norm(irreps_in, squared=squared)
    norms = norm_mod(vecs.reshape(batch, -1))
    norms_true = vecs.norm(dim=-1)
    if squared:
        norms_true.square_()
    assert torch.allclose(norms_true, norms.reshape(batch, n))
