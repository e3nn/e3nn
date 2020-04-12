# pylint: disable=no-member, not-callable, missing-docstring, line-too-long, invalid-name
import argparse

import matplotlib.pyplot as plt
import torch

from e3nn import o3, rsh
from e3nn.util.plot import spherical_surface


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--l_in", type=int, required=True)
    parser.add_argument("--l_out", type=int, required=True)
    parser.add_argument("--n", type=int, default=30, help="size of the SOFT grid")
    parser.add_argument("--dpi", type=float, default=100)
    parser.add_argument("--sep", type=float, default=0.5, help="space between matrices")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    x, y, z, alpha, beta = spherical_surface(args.n)

    out = []
    for l in range(abs(args.l_out - args.l_in), args.l_out + args.l_in + 1):
        C = o3.wigner_3j(args.l_out, args.l_in, l)
        Y = rsh.spherical_harmonics(l, alpha, beta)
        out.append(torch.einsum("ijk,k...->ij...", (C, Y)))
    f = torch.stack(out)

    nf, dim_out, dim_in, *_ = f.size()

    f = 0.5 + 0.5 * f / f.abs().max()

    fig = plt.figure(figsize=(nf * dim_in + (nf - 1) * args.sep, dim_out), dpi=args.dpi)

    for index in range(nf):
        for i in range(dim_out):
            for j in range(dim_in):
                width = 1 / (nf * dim_in + (nf - 1) * args.sep)
                height = 1 / dim_out
                rect = [
                    (index * (dim_in + args.sep) + j) * width,
                    (dim_out - i - 1) * height,
                    width,
                    height
                ]
                ax = fig.add_axes(rect, projection='3d')

                fc = plt.get_cmap("bwr")(f[index, i, j].detach().cpu().numpy())

                ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), rstride=1, cstride=1, facecolors=fc)
                ax.set_axis_off()

                a = 0.6
                ax.set_xlim3d(-a, a)
                ax.set_ylim3d(-a, a)
                ax.set_zlim3d(-a, a)

                ax.view_init(90, 0)

    plt.savefig("kernels{}to{}.png".format(args.l_in, args.l_out), transparent=True)


main()
