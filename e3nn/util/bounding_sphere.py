# pylint: disable=C,R,E1101,E1102
"""
Compute exact minimum bounding sphere of a 3D point cloud using Welzl's algorithm.

code copied from https://github.com/shrx/mbsc

simplified, remove colinearity checks, added naive tests in fit_sphere
"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=E
from scipy.spatial.qhull import QhullError  # pylint: disable=E
from scipy import linalg


def bounding_sphere(array, eps):
    array = np.array(array)
    array = np.unique(array, axis=0)

    try:
        hull = ConvexHull(array)
        array = array[hull.vertices]
    except QhullError:
        pass

    array = np.random.permutation(array)

    if len(array) <= 4:
        return fit_sphere(array, eps)

    return B_min_sphere(list(array), [], eps)


def B_min_sphere(P, B, eps):
    if len(B) == 4 or len(P) == 0:
        return fit_sphere(B, eps)

    # Remove the last (i.e., end) point, p, from the list
    p = P[-1]
    P = P[:-1]

    # Check if p is on or inside the bounding sphere. If not, it must be part of the new boundary.
    radius, center = B_min_sphere(P, B, eps)

    if np.isnan(radius) or np.isinf(radius) or radius < eps or np.linalg.norm(p - center) > (radius + eps):
        radius, center = B_min_sphere(P, B + [p], eps)

    return radius, center


def fit_sphere(array, eps):
    """Fit a sphere to a set of 2, 3, or at most 4 points in 3D space."""

    array = np.array(array)
    N = len(array)

    # Empty set
    if N == 0:
        R = np.nan
        C = np.full(3, np.nan)
        return R, C

    # A single point
    elif N == 1:
        R = 0.
        C = array[0]
        return R, C

    # Line segment
    elif N == 2:
        R = np.linalg.norm(array[1] - array[0]) / 2
        C = np.mean(array, axis=0)
        return R, C

    elif N == 3:
        R, C = fit_sphere(array[[1, 2]], eps)
        if np.linalg.norm(array[0] - C) <= (R + eps):
            return R, C
        R, C = fit_sphere(array[[0, 2]], eps)
        if np.linalg.norm(array[1] - C) <= (R + eps):
            return R, C
        R, C = fit_sphere(array[[0, 1]], eps)
        if np.linalg.norm(array[2] - C) <= (R + eps):
            return R, C

        D12 = array[1] - array[0]
        D12 = D12 / np.linalg.norm(D12)
        D13 = array[2] - array[0]
        D13 = D13 / np.linalg.norm(D13)

        # Make plane formed by the points parallel with the xy-plane
        n = np.cross(D13, D12)
        n = n / np.linalg.norm(n)
        r = np.cross(n, np.array([0, 0, 1]))
        if np.linalg.norm(r) > 0:
            r = np.arccos(n[2]) * r / np.linalg.norm(r)  # Euler rotation vector

        Rmat = linalg.expm(np.array([
            [0., -r[2], r[1]],
            [r[2], 0., -r[0]],
            [-r[1], r[0], 0.]
        ]))

        Xr = np.transpose(np.dot(Rmat, np.transpose(array)))

        # Circle centroid
        x = Xr[:, :2]
        A = 2 * (x[1:] - np.full(2, x[0]))
        b = np.sum((np.square(x[1:]) - np.square(np.full(2, x[0]))), axis=1)
        C = np.transpose(np.linalg.solve(A, b))

        # Circle radius
        R = np.sqrt(np.sum(np.square(x[0] - C)))

        # Rotate centroid back into the original frame of reference
        C = np.append(C, [np.mean(Xr[:, 2])], axis=0)
        C = np.transpose(np.dot(np.transpose(Rmat), C))
        return R, C

    elif N == 4:
        R, C = fit_sphere(array[[1, 2, 3]], eps)
        if np.linalg.norm(array[0] - C) <= (R + eps):
            return R, C
        R, C = fit_sphere(array[[0, 2, 3]], eps)
        if np.linalg.norm(array[1] - C) <= (R + eps):
            return R, C
        R, C = fit_sphere(array[[0, 1, 3]], eps)
        if np.linalg.norm(array[2] - C) <= (R + eps):
            return R, C
        R, C = fit_sphere(array[[0, 1, 2]], eps)
        if np.linalg.norm(array[3] - C) <= (R + eps):
            return R, C

        # Centroid of the sphere
        A = 2 * (array[1:] - np.full(len(array) - 1, array[0]))
        b = np.sum((np.square(array[1:]) - np.square(np.full(len(array) - 1, array[0]))), axis=1)
        C = np.transpose(np.linalg.solve(A, b))

        # Radius of the sphere
        R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))

        return R, C

    else:
        print('Input must a N-by-3 array of point coordinates, with N<=4')
