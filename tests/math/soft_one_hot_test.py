import pytest
import torch

from e3nn.math import soft_one_hot_linspace


@pytest.mark.parametrize('base', ['gaussian', 'cosine', 'fourier'])
def test_zero_out(base):
    x1 = torch.linspace(-2.0, -1.1, 20)
    x2 = torch.linspace(2.1, 3.0, 20)
    x = torch.cat([x1, x2])

    y = soft_one_hot_linspace(x, -1.0, 2.0, 5, base, endpoint=False)
    if base == 'gaussian':
        assert y.abs().max() < 0.22
    else:
        assert y.abs().max() == 0.0


@pytest.mark.parametrize('base', ['gaussian', 'cosine', 'fourier'])
@pytest.mark.parametrize('endpoint', [True, False])
def test_normalized(base, endpoint):
    x = torch.linspace(-14.0, 105.0, 50)
    y = soft_one_hot_linspace(x, -20.0, 120.0, 12, base, endpoint)

    assert 0.5 < y.pow(2).sum(1).min()
    assert y.pow(2).sum(1).max() < 2.0
