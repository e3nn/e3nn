import pytest
import torch

from e3nn.math import soft_one_hot_linspace


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
def test_with_compile(basis) -> None:
    # torch.compile recompiles for every basis and every dtype
    torch._dynamo.config.cache_size_limit = 32

    x = torch.linspace(-2.0, 3.0, 20)
    kwargs = dict(start=-1.0, end=2.0, number=5, basis=basis, cutoff=True)

    y = soft_one_hot_linspace(x, **kwargs)
    y_compiled = torch.compile(soft_one_hot_linspace, fullgraph=True)(x, **kwargs)

    assert y.shape == y_compiled.shape
    assert y.dtype == y_compiled.dtype
    assert y.device == y_compiled.device

    assert torch.allclose(y, y_compiled, atol=1e-7)


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
def test_zero_out(basis) -> None:
    x1 = torch.linspace(-2.0, -1.1, 20)
    x2 = torch.linspace(2.1, 3.0, 20)
    x = torch.cat([x1, x2])

    y = soft_one_hot_linspace(x, -1.0, 2.0, 5, basis, cutoff=True)
    if basis == "gaussian":
        assert y.abs().max() < 0.22
    else:
        assert y.abs().max() == 0.0


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "smooth_finite"])
@pytest.mark.parametrize("cutoff", [True, False])
def test_normalized(basis, cutoff) -> None:
    x = torch.linspace(-14.0, 105.0, 50)
    y = soft_one_hot_linspace(x, -20.0, 120.0, 12, basis, cutoff)

    assert 0.4 < y.pow(2).sum(1).min()
    assert y.pow(2).sum(1).max() < 2.0
