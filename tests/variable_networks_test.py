# pylint: disable=no-member, arguments-differ, redefined-builtin, missing-docstring, line-too-long, invalid-name
import torch

from e3nn import o3
from e3nn import rs
from variable_networks import VariableParityNetwork


def test_variable_parity_network():
    torch.set_default_dtype(torch.float64)

    ends_lmax = 3
    Rs = [(1, l, 1) for l in range(ends_lmax + 1)]
    lmaxes = [3,2,1,2,3]
    muls = [[4,3,2,1],[4,3,2],[4,3],[2,2,2],[1,1,1,1]]
    model = VariableParityNetwork(Rs, Rs, lmaxes=lmaxes, muls=muls,  feature_product=True)

    features = rs.randn(1, 4, Rs)
    geometry = torch.randn(1, 4, 3)

    output = model(features, geometry)

    angles = o3.rand_angles()
    D = rs.rep(Rs, *angles, 1)
    R = -o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

    assert (output - output2).abs().max() < 1e-10 * output.abs().max()

if __name__ == '__main__':
    test_variable_parity_network()