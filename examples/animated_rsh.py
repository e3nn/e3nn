# pylint: disable=not-callable, no-member, invalid-name, missing-docstring, line-too-long
import math
import os
import subprocess
import argparse
import shutil
import tqdm

import plotly.graph_objs as go
import torch

from e3nn import o3, rsh


def rsh_surface(l, m, scale, tr, rot):
    n = 50
    a = torch.linspace(0, 2 * math.pi, 2 * n)
    b = torch.linspace(0, math.pi, n)
    a, b = torch.meshgrid(a, b)

    f = rsh.spherical_harmonics_alpha_beta([l], a, b)
    f = torch.einsum('ij,...j->...i', o3.irr_repr(l, *rot), f)
    f = f[..., l + m]

    r = o3.angles_to_xyz(a, b)
    x, y, z = r[:, :, 0], r[:, :, 1], r[:, :, 2]

    r = f.abs()
    x = scale * r * x + tr[0]
    y = scale * r * y + tr[1]
    z = scale * r * z + tr[2]

    max_value = 0.5

    return go.Surface(
        x=x.numpy(),
        y=y.numpy(),
        z=z.numpy(),
        surfacecolor=f.numpy(),
        showscale=False,
        cmin=-max_value,
        cmax=max_value,
        colorscale=[[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']],
    )


def main(lmax, resolution, steps):
    scale = 0.5 * math.sqrt(4 * math.pi) / math.sqrt(2 * lmax + 1)

    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
        nticks=3,
        range=[-lmax / 2 - 0.5, lmax / 2 + 0.5]
    )

    layout = dict(
        width=resolution,
        height=resolution,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-1.3, z=0),
                projection=dict(type='perspective'),
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    if os.path.exists('sh'):
        shutil.rmtree('sh')
    os.makedirs('sh')

    for i in tqdm.tqdm(range(steps)):
        rot = 2 * math.pi * i / steps
        a, b, c = 0, math.pi / 4, 0
        abc = o3.compose(-c, -b, -a, *o3.compose(0, 0, rot, a, b, c))

        surfaces = [
            rsh_surface(l, m, scale, [l + (m if m < 0 else 0) - lmax / 2, 0, lmax / 2 - l + (m if m > 0 else 0)], abc)
            for l in range(lmax + 1)
            for m in range(-l, l + 1)
        ]

        fig = go.Figure(surfaces, layout=layout)
        fig.write_image('sh/{:03d}.png'.format(i))

    subprocess.check_output(["convert", "-delay", "3", "-loop", "0", "-dispose", "2", "sh/*.png", "output.gif"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30)

    args = parser.parse_args()

    main(args.lmax, args.resolution, args.steps)
