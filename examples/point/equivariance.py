from functools import partial

import torch

from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.non_linearities.rescaled_act import relu, sigmoid, tanh, absolute
from se3cnn.non_linearities.gated_block_parity import GatedBlockParity
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import ConstantRadialModel
from se3cnn.SO3 import normalizeRs, rep, rot


def check_rotation():
    pass


def check_rotation_parity():
    K = partial(Kernel, RadialModel=ConstantRadialModel)
    C = partial(Convolution, K)

    Rs_in = [(1, 0, 1)]
    m = GatedBlockParity(
        C,
        Rs_in,
        [(4, 0, 1)],
        [(-1, relu)],
        [(8, 0, 1)],
        [(-1, sigmoid)],
        [(4, 1, -1), (4, 2, 1)]
    )
    # Rs_in and m.Rs_out


def check_network_equivariance(network, Rs_in, Rs_out, vector_output: bool = False, batch: int = 2,
                               n_atoms: int = 6) -> bool:
    """
    from user func(input_data, alpha, beta, gamma, parity)
    from user func(output_data,  alpha, beta, gamma, parity)

    Checks rotation equivariance for a function. Network output transforms with the wigner d matrices.

    :param network: Often the forward pass of an object inheriting from torch.nn.Module
    :param Rs_in: Rs for the input of the network.
    :param Rs_out: Rs for the output of the network.
    :param vector_output: When True, the network outputs vectors which have already been converted to xyz basis.
    :param batch: Batch size of test case.
    :param n_atoms: Number of atoms of test case.
    :return: None
    """

    def check_parity(Rs):
        """Does Rs require a test of parity equivariance."""
        if any(p != 0 for _, _, p in normalizeRs(Rs)):
            return True
        else:
            return False

    # Make sure that all parameters are on the same device. Set that to calculation device.
    devices = [i.device for i in network.parameters()]
    device = devices[0]
    assert all([device == i for i in devices])

    Rs_in = normalizeRs(Rs_in)
    Rs_out = normalizeRs(Rs_out)
    parity = check_parity(Rs_in) or check_parity(Rs_out)

    abc = torch.randn(3, device=device)  # Rotation seed of euler angles.
    D_in = rep(Rs_in, *abc)
    geo_rotation_matrix = -rot(*abc) if parity else rot(*abc)  # Geometry is odd parity.
    D_out = rep(Rs_out, *abc)

    c = sum([mul * (2 * l + 1) for mul, l, _ in Rs_in])
    feat = torch.randn(batch, n_atoms, c, device=device)  # Transforms with wigner D matrix
    geo = torch.randn(batch, n_atoms, 3, device=device)  # Transforms with rotation matrix.

    F = network(feat, geo)
    if vector_output:
        RF = torch.einsum("ij,zkj->zki", geo_rotation_matrix, F)
    else:
        RF = torch.einsum("ij,zkj->zki", D_out, F)
    FR = network(feat @ D_in.t(), geo @ geo_rotation_matrix.t())  # [batch, feat, N]
    return (RF - FR).norm() < 10e-5 * RF.norm()
