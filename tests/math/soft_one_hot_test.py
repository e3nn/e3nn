import pytest
import torch

from e3nn.math import soft_one_hot_linspace


@pytest.mark.parametrize('basis', ['gaussian', 'cosine', 'fourier', 'bessel', 'smooth_finite'])
def test_zero_out(basis):
    x1 = torch.linspace(-2.0, -1.1, 20)
    x2 = torch.linspace(2.1, 3.0, 20)
    x = torch.cat([x1, x2])

    y = soft_one_hot_linspace(x, -1.0, 2.0, 5, basis, cutoff=True)
    if basis == 'gaussian':
        assert y.abs().max() < 0.22
    else:
        assert y.abs().max() == 0.0


@pytest.mark.parametrize('basis', ['gaussian', 'cosine', 'fourier', 'smooth_finite'])
@pytest.mark.parametrize('cutoff', [True, False])
def test_normalized(basis, cutoff):
    x = torch.linspace(-14.0, 105.0, 50)
    y = soft_one_hot_linspace(x, -20.0, 120.0, 12, basis, cutoff)

    assert 0.4 < y.pow(2).sum(1).min()
    assert y.pow(2).sum(1).max() < 2.0
