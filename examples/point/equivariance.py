# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.non_linearities.gated_block_parity import GatedBlockParity
from e3nn.non_linearities.rescaled_act import absolute, relu, sigmoid, tanh
from e3nn.point.operations import Convolution
from e3nn.radial import ConstantRadialModel


def check_rotation(batch: int = 10, n_atoms: int = 25):
    # Setup the network.
    K = partial(Kernel, RadialModel=ConstantRadialModel)
    Rs_in = [(1, 0), (1, 1)]
    Rs_out = [(1, 0), (1, 1), (1, 2)]
    act = GatedBlock(
        Rs_out,
        scalar_activation=sigmoid,
        gate_activation=absolute,
    )
    conv = Convolution(K, Rs_in, act.Rs_in)

    # Setup the data. The geometry, input features, and output features must all rotate.
    abc = torch.randn(3)  # Rotation seed of euler angles.
    rot_geo = o3.rot(*abc)
    D_in = rs.rep(Rs_in, *abc)
    D_out = rs.rep(Rs_out, *abc)

    feat = torch.randn(batch, n_atoms, rs.dim(Rs_in))  # Transforms with wigner D matrix
    geo = torch.randn(batch, n_atoms, 3)  # Transforms with rotation matrix.

    # Test equivariance.
    F = act(conv(feat, geo))
    RF = torch.einsum("ij,zkj->zki", D_out, F)
    FR = act(conv(feat @ D_in.t(), geo @ rot_geo.t()))
    return (RF - FR).norm() < 10e-5 * RF.norm()


def check_rotation_parity(batch: int = 10, n_atoms: int = 25):
    # Setup the network.
    K = partial(Kernel, RadialModel=ConstantRadialModel)
    Rs_in = [(1, 0, +1)]
    act = GatedBlockParity(
        Rs_scalars=[(4, 0, +1)],
        act_scalars=[(-1, relu)],
        Rs_gates=[(8, 0, +1)],
        act_gates=[(-1, tanh)],
        Rs_nonscalars=[(4, 1, -1), (4, 2, +1)]
    )
    conv = Convolution(K, Rs_in, act.Rs_in)
    Rs_out = act.Rs_out

    # Setup the data. The geometry, input features, and output features must all rotate and observe parity.
    abc = torch.randn(3)  # Rotation seed of euler angles.
    rot_geo = -o3.rot(*abc)  # Negative because geometry has odd parity. i.e. improper rotation.
    D_in = rs.rep(Rs_in, *abc, parity=1)
    D_out = rs.rep(Rs_out, *abc, parity=1)

    feat = torch.randn(batch, n_atoms, rs.dim(Rs_in))  # Transforms with wigner D matrix and parity.
    geo = torch.randn(batch, n_atoms, 3)  # Transforms with rotation matrix and parity.

    # Test equivariance.
    F = act(conv(feat, geo))
    RF = torch.einsum("ij,zkj->zki", D_out, F)
    FR = act(conv(feat @ D_in.t(), geo @ rot_geo.t()))
    return (RF - FR).norm() < 10e-5 * RF.norm()


if __name__ == '__main__':
    check_rotation()
    check_rotation_parity()
