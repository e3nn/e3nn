import pytest

from typing import Optional

import torch

from e3nn import experimental, o3
from e3nn.util.test import assert_equivariant, random_irreps, assert_normalized


def test_zero_dim():
    tp = experimental.TensorProduct(
        "0x0e + 1e",
        "0e + 0x1e",
        "0x0e + 1e",
        [
            (0, 0, 0, "uvw", True),
            (1, 1, 0, "uvw", True),
        ],
    )
    w = [torch.randn(ins.path_shape) for ins in tp.instructions if ins.has_weight]
    x = tp.irreps_in1.randn(-1)
    y = tp.irreps_in2.randn(-1)

    print(tp(x, y, w).shape)
