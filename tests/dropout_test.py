# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name
import pytest
import torch
from e3nn import rs
from e3nn.dropout import Dropout


@pytest.mark.parametrize(
    "Rs, p, dtype",
    [([0, 0, 1, (2, 2), 4], 0.5, torch.float64), ([0, (2, 1), 2], 0.75, torch.float32)],
)
def test_that_it_runs(Rs, p, dtype):
    x = rs.randn(10, Rs, dtype=dtype)
    m = Dropout(Rs, p=p)

    m.train(True)
    y = m(x)
    assert ((y == x / (1 - p)) | (y == 0)).all()

    m.train(False)
    assert (m(x) == x).all()
