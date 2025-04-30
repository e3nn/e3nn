import torch
import pytest
import e3nn

@pytest.mark.parametrize("n", [1, 2, 4])
def test_bessel(n: int):
    x = torch.linspace(0.0, 1.0, 100)
    y = e3nn.math.bessel(x, n)
    assert y.shape == (100, n)
    
    