# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
from functools import partial

import pytest
import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.point.message_passing import Convolution, SmallConvolution
from e3nn.radial import ConstantRadialModel


@pytest.mark.parametrize('Rs_in, Rs_out, n_source, n_target, n_edge', itertools.product([[1]], [[2]], [2, 3], [1, 3], [0, 3]))
def test_equivariance(Rs_in, Rs_out, n_source, n_target, n_edge):
    torch.set_default_dtype(torch.float64)

    mp = Convolution(Kernel(Rs_in, Rs_out, ConstantRadialModel))
    mp_small = SmallConvolution(partial(Kernel, RadialModel=ConstantRadialModel), Rs_in, Rs_out, lin_mul=2, tp_mul=1)

    features = rs.randn(n_target, Rs_in)

    r_source = torch.randn(n_source, 3)
    r_target = torch.randn(n_target, 3)

    edge_index = torch.stack([
        torch.randint(n_source, size=(n_edge,)),
        torch.randint(n_target, size=(n_edge,)),
    ])
    size = (n_target, n_source)

    if n_edge == 0:
        edge_r = torch.zeros(0, 3)
    else:
        edge_r = torch.stack([
            r_target[j] - r_source[i]
            for i, j in edge_index.T
        ])
    print(features.shape, edge_index.shape, edge_r.shape, size)
    out1 = mp(features, edge_index, edge_r, size=size)
    out_small1 = mp_small(features, edge_index, edge_r, size=size)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles)
    D_out = rs.rep(Rs_out, *angles)
    R = o3.rot(*angles)

    out2 = mp(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out
    out_small2 = mp_small(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out

    assert (out1 - out2).abs().max() < 1e-10
    assert (out_small1 - out_small2).abs().max() < 1e-10


def test_flow():
    """
    This test checks that information is flowing as expected from target to source.
    edge_index[0] is source (convolution center)
    edge_index[1] is target (neighbors)
    """

    edge_index = torch.LongTensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    features = torch.tensor(
        [-1., 1., 1., 1., 1.]
    )
    features = features.unsqueeze(-1)
    edge_r = torch.ones(edge_index.shape[-1], 3)

    Rs = [0]
    conv = Convolution(Kernel(Rs, Rs, ConstantRadialModel))
    conv.kernel.R.weight.data.fill_(1.)  # Fix weight to 1.

    output = conv(features, edge_index, edge_r)
    torch.allclose(output, torch.tensor([4., 0., 0., 0., 0.]).unsqueeze(-1))

    edge_index = torch.LongTensor([
        [1, 2, 3, 4],
        [0, 0, 0, 0]
    ])
    output = conv(features, edge_index, edge_r)
    torch.allclose(output, torch.tensor([0., -1., -1., -1., -1.]).unsqueeze(-1))
