# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, protected-access
import torch

from e3nn.image.filter import LowPassFilter


def test_low_pass_filter():
    x = torch.randn(3, 3, 32, 32, 32, 2)
    x = LowPassFilter(2.0, 2)(x)
    assert x.shape == (3, 3, 16, 16, 16, 2)
