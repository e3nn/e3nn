# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from functools import partial

from e3nn import o3, rs
from e3nn.kernel import Kernel, GroupKernel
from e3nn.point.message_passing import Convolution
from e3nn.radial import ConstantRadialModel
from e3nn.point.depthwise import DepthwiseConvolution, DepthwiseConvolutionParity


def test_equivariance():
    torch.set_default_dtype(torch.float64)

    n_edge = 3
    n_source = 4
    n_target = 2

    Rs_in = [(3, 0), (0, 1)]
    Rs_mid1 = [(5, 0), (1, 1)]
    Rs_mid2 = [(5, 0), (1, 1), (1, 2)]
    Rs_out = [(5, 1), (3, 2)]

    Rs_in_p = [(3, 0, 1), (0, 1, -1)]
    Rs_mid1_p = [(5, 0, 1), (1, 1, -1)]
    Rs_mid2_p = [(5, 0, 1), (1, 1, -1), (1, 2, 1)]
    Rs_out_p = [(5, 1, -1), (3, 2, 1)]

    convolution = lambda Rs_in, Rs_out: Convolution(Kernel(Rs_in, Rs_out, ConstantRadialModel))
    convolution_groups = lambda Rs_in, Rs_out: Convolution(
        GroupKernel(Rs_in, Rs_out, partial(Kernel, RadialModel=ConstantRadialModel), groups))

    groups = 4
    mp = DepthwiseConvolution(Rs_in, Rs_out, Rs_mid1, Rs_mid2, groups, convolution)
    mp_p = DepthwiseConvolutionParity(Rs_in_p, Rs_out_p, Rs_mid1_p, Rs_mid2_p, groups, convolution)
    mp_groups = DepthwiseConvolution(Rs_in, Rs_out, Rs_mid1, Rs_mid2, groups, convolution_groups)

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
    out1_p = mp_p(features, edge_index, edge_r, size=size)
    out1_groups = mp_groups(features, edge_index, edge_r, size=size)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles)
    D_out = rs.rep(Rs_out, *angles)
    R = o3.rot(*angles)

    out2 = mp(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out
    print('out2', out2.shape)
    out2_p = mp_p(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out
    print('out2_p', out2_p.shape)
    out2_groups = mp_groups(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out

    assert (out1 - out2).abs().max() < 1e-10
    assert (out1_p - out2_p).abs().max() < 1e-10
    assert (out1_groups - out2_groups).abs().max() < 1e-10
