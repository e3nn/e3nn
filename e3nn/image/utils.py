# pylint: disable=C,E1101
import numpy as np
from scipy.ndimage import affine_transform


def rotate_scalar(x, rot):
    # this function works in the convention field[x, y, z]
    invrot = np.linalg.inv(rot)
    center = (np.array(x.shape) - 1) / 2
    return affine_transform(x, matrix=invrot, offset=center - np.dot(invrot, center))


def rotate_field(x, rot, R):
    # this function works in the convention field[x, y, z]
    invrot = np.linalg.inv(rot)
    y = np.empty_like(x)
    for k in range(y.shape[0]):
        center = (np.array(x.shape[1:]) - 1) / 2
        y[k] = affine_transform(x[k], matrix=invrot, offset=center - np.dot(invrot, center))

    return np.einsum("ij,j...->i...", R, y)
