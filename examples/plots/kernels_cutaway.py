# pylint: disable=C,R,E1101
from se3cnn import kernel
from se3cnn import SO3
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def plot_kernel(basis, base_element=0, zheight=0):
    size = basis.shape[-1]
    dim_out = basis.shape[1]
    dim_in = basis.shape[2]

    vmin = basis.mean() - 2 * basis.std()
    vmax = basis.mean() + 2 * basis.std()

    plt.figure(figsize=(2*dim_in, 2*dim_out))
    for i in range(dim_out):
        for j in range(dim_in):
            plt.subplot(dim_out, dim_in, dim_in * i + j + 1)
            plt.imshow(basis[base_element, i, j, size//2 + round(size / 2 * zheight), :, :], vmin=vmin, vmax=vmax)
            plt.axis("off")
    plt.tight_layout()


def main():
    # Render a big kernel
    size = 13
    R_in = 2
    R_out = 2

    window = partial(kernel.gaussian_window, radii=[size / 4], J_max_list=[100], sigma=size / 8)
    basis = kernel.cube_basis_kernels(size, R_in, R_out, window)

    print(basis.shape)

    # Check equivariance
    print(kernel.check_basis_equivariance(basis, R_in, R_out, 3.14/4, 0.12, 0.05))

    for i in range(basis.shape[0]):
        print(i)
        plot_kernel(basis, base_element=i)
        plt.show()


main()
