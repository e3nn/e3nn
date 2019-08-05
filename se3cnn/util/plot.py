# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring
import math

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

from se3cnn.SO3 import spherical_harmonics, angles_to_xyz


def spherical_surface(n):
    beta = torch.linspace(1e-16, math.pi - 1e-16, 2 * n)
    alpha = torch.linspace(0, 2 * math.pi, 2 * n)
    beta_, alpha_ = torch.meshgrid(beta, alpha)
    x, y, z = angles_to_xyz(beta_, alpha_)

    beta = 0.5 * (beta[1:] + beta[:-1])
    alpha = 0.5 * (alpha[1:] + alpha[:-1])
    beta, alpha = torch.meshgrid(beta, alpha)
    return x, y, z, alpha, beta


def spherical_harmonic_signal(coeff, alpha, beta):
    from itertools import count
    s = 0
    i = 0
    for l in count():
        d = 2 * l + 1
        if len(coeff) < i + d:
            break
        c = coeff[i: i + d]
        i += d

        s += torch.einsum("i,i...->...", (c, spherical_harmonics(l, alpha, beta)))
    return s


def plot_sh_signal(coeff, n=20):
    from functools import partial

    fun = partial(spherical_harmonic_signal, coeff)
    plot_sphere(fun, n)


def plot_sphere(fun, n=20):
    """
    :param fun: function of (alpha, beta)
    :param n: precision
    """
    x, y, z, a, b = spherical_surface(n)

    f = fun(a, b)
    f = 0.5 + 0.5 * f.div(f.abs().max())  # get a signal in the interval [0, 1]
    fc = plt.get_cmap("bwr")(f.detach().cpu().numpy())

    ax = plt.gca(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), rstride=1, cstride=1, facecolors=fc)

    ax.set_axis_off()
    a = 0.6
    ax.set_xlim3d(-a, a)
    ax.set_ylim3d(-a, a)
    ax.set_zlim3d(-a, a)

    ax.view_init(90, 0)


def plotly_sphere(fun, n=240, radius=False):
    """
    surface = plotly_sphere(partial(spherical_harmonic_signal, x))
    fig = go.Figure(data=[surface])
    fig.show()
    """
    import plotly.graph_objs as go

    a = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    b = torch.linspace(0, math.pi, n, dtype=torch.float64)
    a, b = torch.meshgrid(a, b)

    f = fun(a, b)
    x, y, z = angles_to_xyz(a, b)

    if radius:
        r = f.abs()
        x *= r
        y *= r
        z *= r

    return go.Surface(x=x.numpy(), y=y.numpy(), z=z.numpy(), surfacecolor=f.numpy())
