# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, protected-access
import pytest
import torch

from e3nn import rs
from e3nn.image.convolution import Convolution


@pytest.mark.parametrize('fuzzy_pixels', [False, True])
def test_equivariance(fuzzy_pixels):
    torch.set_default_dtype(torch.float64)

    f = torch.nn.Sequential(
        Convolution(
            Rs_in=[0],
            Rs_out=[0, 0, 1, 1, 2],
            size=5,
            steps=(0.5, 0.5, 0.9),
            fuzzy_pixels=fuzzy_pixels
        ),
        Convolution(
            Rs_in=[0, 0, 1, 1, 2],
            Rs_out=[0],
            size=5,
            steps=(0.5, 0.5, 0.9),
            fuzzy_pixels=fuzzy_pixels
        ),
    )

    def rotate(t):
        # rotate 90 degrees in plane of axes 1 and 2
        return t.flip(1).transpose(1, 2)

    def unrotate(t):
        # undo the rotation by 3 more rotations
        return rotate(rotate(rotate(t)))

    inp = torch.randn(2, 16, 16, 16, 1)
    inp_r = rotate(inp)

    diff_inp = (inp - unrotate(inp_r)).abs().max().item()
    assert diff_inp < 1e-10  # sanity check

    out = f(inp)
    out_r = f(inp_r)

    diff_out = (out - unrotate(out_r)).abs().max().item()
    assert diff_out < 1e-1 if fuzzy_pixels else 1e-10


@pytest.mark.parametrize('fuzzy_pixels', [False, True])
def test_normalization(fuzzy_pixels):
    batch = 3
    size = 5
    input_size = 15
    Rs_in = [(20, 0), (20, 1), (10, 2)]
    Rs_out = [0, 1, 2]

    conv = Convolution(Rs_in, Rs_out, size, lmax=2, fuzzy_pixels=fuzzy_pixels)

    x = rs.randn(batch, input_size, input_size, input_size, Rs_in)
    y = conv(x)

    assert y.shape[-1] == rs.dim(Rs_out)
    assert y.var().log10().abs() < 1.5
