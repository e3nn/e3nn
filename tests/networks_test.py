# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
import torch

from e3nn import o3
from e3nn import rs
from e3nn.networks import (
    GatedConvParityNetwork,
    GatedConvNetwork,
    ImageS2Network,
    S2ConvNetwork,
    S2ParityNetwork,
    DepthwiseNetwork,
)


def test_parity_network():
    torch.set_default_dtype(torch.float64)

    lmax = 3
    Rs = [(1, l, 1) for l in range(lmax + 1)]
    model = GatedConvParityNetwork(Rs, 4, Rs, lmax, feature_product=True)

    features = rs.randn(1, 4, Rs)
    geometry = torch.randn(1, 4, 3)

    output = model(features, geometry)

    angles = o3.rand_angles()
    D = rs.rep(Rs, *angles, 1)
    R = -o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

    assert (output - output2).abs().max() < 1e-10 * output.abs().max()


def test_network():
    torch.set_default_dtype(torch.float64)

    lmax = 3
    Rs = [(1, l) for l in range(lmax + 1)]
    model = GatedConvNetwork(Rs, 4 * Rs, Rs, lmax, feature_product=True)

    features = rs.randn(1, 4, Rs)
    geometry = torch.randn(1, 4, 3)

    output = model(features, geometry)

    angles = o3.rand_angles()
    D = rs.rep(Rs, *angles)
    R = o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

    assert (output - output2).abs().max() < 1e-10 * output.abs().max()


def test_image_network():
    torch.set_default_dtype(torch.float64)

    Rs = [0, 0, 3]

    model = ImageS2Network(
        Rs_in=Rs,
        mul=4,
        lmax=6,
        Rs_out=Rs,
        size=5,
        layers=3
    )

    image = rs.randn(1, 16, 16, 16, Rs)
    model(image)


def test_s2conv_network():
    torch.set_default_dtype(torch.float64)

    lmax = 3
    Rs = [(1, l, 1) for l in range(lmax + 1)]
    model = S2ConvNetwork(Rs, 4, Rs, lmax)

    features = rs.randn(1, 4, Rs)
    geometry = torch.randn(1, 4, 3)

    output = model(features, geometry)

    angles = o3.rand_angles()
    D = rs.rep(Rs, *angles, 1)
    R = -o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

    assert (output - output2).abs().max() < 1e-10 * output.abs().max()


def test_equivariance_s2parity_network():
    torch.set_default_dtype(torch.float64)
    mul = 3
    Rs_in = [(mul, l, -1) for l in range(3 + 1)]
    Rs_out = [(mul, l, 1) for l in range(3 + 1)]

    net = S2ParityNetwork(Rs_in, mul, lmax=3, Rs_out=Rs_out)

    abc = o3.rand_angles()
    D_in = rs.rep(Rs_in, *abc, 1)
    D_out = rs.rep(Rs_out, *abc, 1)

    fea = rs.randn(10, Rs_in)

    x1 = torch.einsum("ij,zj->zi", D_out, net(fea))
    x2 = net(torch.einsum("ij,zj->zi", D_in, fea))
    assert (x1 - x2).norm() < 1e-3 * x1.norm()


def test_equivariance_depthwise_network():
    from functools import partial
    from e3nn.radial import ConstantRadialModel
    from e3nn.kernel import Kernel, GroupKernel
    from e3nn.point.message_passing import Convolution
    torch.set_default_dtype(torch.float64)

    Rs_in = [(3, 0, 1), (2, 1, -1)]
    Rs_out = [(2, 2, 1)]
    Rs_hidden = [(5, 0, 1), (5, 0, -1), (3, 1, 1), (3, 1, -1), (3, 2, 1), (3, 2, -1)]
    groups = 4
    lmax = max([l for m, l, p in Rs_hidden])

    kernel = partial(Kernel, RadialModel=ConstantRadialModel,
                     selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax),
                     allow_unused_inputs=True)
    K = lambda Rs_in, Rs_out: GroupKernel(Rs_in, Rs_out, kernel, groups)
    convolution = lambda Rs_in, Rs_out: Convolution(K(Rs_in, Rs_out))

    model = DepthwiseNetwork(Rs_in, Rs_out, Rs_hidden, convolution, groups)

    N = 7
    n_norm = 1
    features = torch.randn(N, rs.dim(Rs_in))
    edge_index = torch.randint(low=0, high=N, size=(2, 4))
    edge_attr = torch.rand(edge_index.shape[-1], 3)

    output = model(features, edge_index, edge_attr, n_norm=n_norm)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles, 1)
    D_out = rs.rep(Rs_out, *angles, 1)
    R = -o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,aj->ai', D_out.T, model(ein('ij,aj->ai', D_in, features), edge_index, ein('ij,aj->ai', R, edge_attr), n_norm=n_norm))
    # Something is wrong with the network because it often will output all zeros for random input data.
    print(output.max(), output2.max())
    print((output - output2).abs().max())
    assert (output - output2).abs().max() < 1e-10 * output.abs().max()
