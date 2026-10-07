import math

import numpy as np
import torch

from e3nn import o3


def hamilton_matrix(q):
    # Independent quaternion group action q*(0,v)*q-conjugate on the three
    # coordinate unit vectors, without the implementation's matrix formula.
    w, xyz = q[0], q[1:]
    columns = []
    for v in np.eye(3):
        left_real = -float(xyz @ v)
        left_vector = w * v + np.cross(xyz, v)
        columns.append(-left_real * xyz + w * left_vector - np.cross(left_vector, xyz))
    return np.stack(columns, axis=-1)


def test_exact_half_turn_matrices_keep_unit_quaternions_and_rotation_axes():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    for dtype in [torch.float32, torch.float64]:
        for axis in axes:
            rotation = torch.tensor(2 * np.outer(axis, axis) - np.eye(3), dtype=dtype)
            quaternion = o3.matrix_to_quaternion(rotation)
            assert torch.isfinite(quaternion).all()
            torch.testing.assert_close(quaternion.norm(), torch.tensor(1.0, dtype=dtype))
            np.testing.assert_allclose(
                hamilton_matrix(quaternion.detach().numpy()), rotation.numpy(), atol=3e-7 if dtype == torch.float32 else 1e-14
            )
            returned_axis, angle = o3.matrix_to_axis_angle(rotation)
            torch.testing.assert_close(angle, torch.tensor(math.pi, dtype=dtype))
            torch.testing.assert_close(returned_axis.norm(), torch.tensor(1.0, dtype=dtype))
            np.testing.assert_allclose(
                np.outer(returned_axis.numpy(), returned_axis.numpy()),
                np.outer(axis, axis),
                atol=3e-7 if dtype == torch.float32 else 1e-14,
            )


def test_quaternion_to_matrix_identity_has_the_analytic_tangent_jacobian():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(o3.quaternion_to_matrix, q)
    generators = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    expected = torch.cat([torch.zeros(3, 3, 1, dtype=torch.float64), 2 * generators.movedim(0, -1)], dim=-1)
    torch.testing.assert_close(jacobian, expected, rtol=0, atol=1e-14)


def test_quaternion_curve_identity_first_and_second_rotation_differentials():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def function(theta):
        return o3.quaternion_to_matrix(o3.axis_angle_to_quaternion(axis, theta))

    first = torch.autograd.functional.jacobian(function, zero)
    expected_first = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(first, expected_first, rtol=0, atol=1e-14)
    for row in range(3):
        for column in range(3):
            second = torch.autograd.functional.hessian(lambda theta: function(theta)[row, column], zero)
            torch.testing.assert_close(second, (expected_first @ expected_first)[row, column], rtol=0, atol=1e-14)


def test_known_ordinary_rotations_batches_empty_and_antipodal_quaternions():
    axes = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 0.0], [-2.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.array([0.4, 1.2, -2.4, 3.0])
    quaternions = np.concatenate([np.cos(angles[:, None] / 2), axes * np.sin(angles[:, None] / 2)], axis=-1)
    rotations = np.stack([hamilton_matrix(q) for q in quaternions])
    for shape in [(4,), (2, 2)]:
        matrix = torch.tensor(rotations.reshape(*shape, 3, 3), dtype=torch.float64)
        q = o3.matrix_to_quaternion(matrix)
        torch.testing.assert_close(
            q, torch.tensor(quaternions.reshape(*shape, 4), dtype=torch.float64), rtol=1e-13, atol=1e-13
        )
        torch.testing.assert_close(o3.quaternion_to_matrix(q), matrix, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(o3.quaternion_to_matrix(-q), matrix, rtol=1e-13, atol=1e-13)
    empty = torch.empty((0, 3, 3), dtype=torch.float64)
    assert o3.matrix_to_quaternion(empty).shape == (0, 4)


def test_actual_first_and_second_gradient_checks_at_identity():
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    # A valid smooth unit-quaternion chart, avoiding perturbations off S^3.
    def function(v):
        q = torch.cat([torch.ones(1, dtype=v.dtype), v])
        q = q / q.norm()
        return o3.quaternion_to_matrix(q)

    assert torch.autograd.gradcheck(function, (zero,))
    assert torch.autograd.gradgradcheck(function, (zero,))


def _matrix_quaternion_from_exponential_coordinates(v):
    # Generic matrix exponential generates valid SO(3) curves independently
    # of the quaternion-to-matrix and axis-angle implementation under test.
    zero = v.new_zeros(())
    x, y, z = v.unbind()
    hat = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    return o3.matrix_to_quaternion(torch.matrix_exp(2 * hat))


def test_matrix_to_quaternion_has_analytic_identity_chart_differentials():
    v = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    # q(v) = [cos(||v||), sinc(||v||) v] in this local chart.
    jacobian = torch.autograd.functional.jacobian(_matrix_quaternion_from_exponential_coordinates, v)
    expected = torch.cat([torch.zeros(1, 3, dtype=v.dtype), torch.eye(3, dtype=v.dtype)], dim=0)
    torch.testing.assert_close(jacobian, expected, rtol=0, atol=1e-14)
    for index in range(4):
        hessian = torch.autograd.functional.hessian(
            lambda value: _matrix_quaternion_from_exponential_coordinates(value)[index], v
        )
        expected = -torch.eye(3, dtype=v.dtype) if index == 0 else torch.zeros(3, 3, dtype=v.dtype)
        torch.testing.assert_close(hessian, expected, rtol=0, atol=1e-14)


def test_matrix_to_quaternion_real_gradient_checks_on_valid_rotation_curves():
    for values in [[0.0, 0.0, 0.0], [0.2, -0.3, 0.1]]:
        v = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(_matrix_quaternion_from_exponential_coordinates, (v,))
        assert torch.autograd.gradgradcheck(_matrix_quaternion_from_exponential_coordinates, (v,))
