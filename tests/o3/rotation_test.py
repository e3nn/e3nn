import torch

from e3nn import o3


def test_xyz(float_tolerance):
    R = o3.rand_matrix(10)
    assert (R @ R.transpose(-1, -2) - torch.eye(3)).abs().max() < float_tolerance

    a, b, c = o3.matrix_to_angles(R)
    pos1 = o3.angles_to_xyz(a, b)
    pos2 = R @ torch.tensor([0, 0, 1.0])
    assert torch.allclose(pos1, pos2)

    a2, b2 = o3.xyz_to_angles(pos2)
    assert (a - a2).abs().max() < float_tolerance
    assert (b - b2).abs().max() < float_tolerance


def test_conversions(float_tolerance):
    def wrap(f):
        def g(x):
            if isinstance(x, tuple):
                return f(*x)
            else:
                return f(x)
        return g

    def identity(x):
        return x
    conv = [
        [identity, wrap(o3.angles_to_matrix), wrap(o3.angles_to_axis_angle), wrap(o3.angles_to_quaternion)],
        [wrap(o3.matrix_to_angles), identity, wrap(o3.matrix_to_axis_angle), wrap(o3.matrix_to_quaternion)],
        [wrap(o3.axis_angle_to_angles), wrap(o3.axis_angle_to_matrix), identity, wrap(o3.axis_angle_to_quaternion)],
        [wrap(o3.quaternion_to_angles), wrap(o3.quaternion_to_matrix), wrap(o3.quaternion_to_axis_angle), identity],
    ]

    R1 = o3.rand_matrix(100)
    path = [1, 2, 3, 0, 2, 0, 3, 1, 3, 2, 1, 0, 1]

    g = R1
    for i, j in zip(path, path[1:]):
        g = conv[i][j](g)
    R2 = g

    assert (R1 - R2).abs().max() < 10*float_tolerance


def test_compose(float_tolerance):
    q1 = o3.rand_quaternion(10)
    q2 = o3.rand_quaternion(10)

    q = o3.compose_quaternion(q1, q2)

    R1 = o3.quaternion_to_matrix(q1)
    R2 = o3.quaternion_to_matrix(q2)

    R = R1 @ R2

    abc1 = o3.quaternion_to_angles(q1)
    abc2 = o3.quaternion_to_angles(q2)

    abc = o3.compose_angles(*abc1, *abc2)

    ax1, a1 = o3.quaternion_to_axis_angle(q1)
    ax2, a2 = o3.quaternion_to_axis_angle(q2)

    ax, a = o3.compose_axis_angle(ax1, a1, ax2, a2)

    R1 = o3.quaternion_to_matrix(q)
    R2 = R
    R3 = o3.angles_to_matrix(*abc)
    R4 = o3.axis_angle_to_matrix(ax, a)

    assert (R1 - R2).norm(dim=1).max() < float_tolerance
    assert (R1 - R3).norm(dim=1).max() < float_tolerance
    assert (R1 - R4).norm(dim=1).max() < float_tolerance
