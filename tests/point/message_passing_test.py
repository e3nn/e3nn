# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import pytest
import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.point.message_passing import MessagePassing
from e3nn.radial import ConstantRadialModel


@pytest.mark.parametrize('Rs_in, Rs_out, n_source, n_target, n_edge', itertools.product([[1]], [[2]], [2, 3], [1, 3], [0, 3]))
def test_E3conv(Rs_in, Rs_out, n_source, n_target, n_edge):
    torch.set_default_dtype(torch.float64)

    mp = MessagePassing(Kernel(Rs_in, Rs_out, ConstantRadialModel))

    features = rs.randn(n_source, Rs_in)

    r_source = torch.randn(n_source, 3)
    r_target = torch.randn(n_target, 3)

    edge_index = torch.stack([
        torch.randint(n_target, size=(n_edge,)),
        torch.randint(n_source, size=(n_edge,)),
    ])
    size = (n_target, n_source)

    if n_edge == 0:
        edge_r = torch.zeros(0, 3)
    else:
        edge_r = torch.stack([
            r_source[j] - r_target[i]
            for i, j in edge_index.T
        ])

    out1 = mp(features, edge_index, edge_r, size)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles)
    D_out = rs.rep(Rs_out, *angles)
    R = o3.rot(*angles)

    out2 = mp(features @ D_in.T, edge_index, edge_r @ R.T, size) @ D_out

    assert (out1 - out2).abs().max() < 1e-10
