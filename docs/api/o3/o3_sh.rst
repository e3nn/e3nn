o3 - Spherical Harmonics
========================

.. jupyter-execute::
    :hide-code:

    import torch
    import math
    from e3nn import o3
    import plotly.graph_objects as go

    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
        nticks=3,
    )

    layout = dict(
        width=690,
        height=160,
        scene=dict(
            xaxis=dict(
                **axis,
                range=[-8, 8]
            ),
            yaxis=dict(
                **axis,
                range=[-2, 2]
            ),
            zaxis=dict(
                **axis,
                range=[-2, 2]
            ),
            aspectmode='manual',
            aspectratio=dict(x=8, y=2, z=2),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-5, z=5),
                projection=dict(type='orthographic'),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    cmap_bwr = [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]

    def s2_grid():
        betas = torch.linspace(0, math.pi, 50)
        alphas = torch.linspace(0, 2 * math.pi, 100)
        beta, alpha = torch.meshgrid(betas, alphas)
        return o3.angles_to_xyz(alpha, beta)

    def trace(r, f, c):
        return dict(
            x=f.abs() * r[..., 0] + c[0],
            y=f.abs() * r[..., 1] + c[1],
            z=f.abs() * r[..., 2] + c[2],
            surfacecolor=f
        )

    def plot(data):
        n = data.shape[-1]
        traces = [
            trace(r, data[..., i], torch.tensor([2.0 * i - (n - 1.0), 0.0, 0.0]))
            for i in range(n)
        ]
        cmax = max(d['surfacecolor'].abs().max().item() for d in traces)
        traces = [go.Surface(**d, colorscale=cmap_bwr, cmin=-cmax, cmax=cmax) for d in traces]
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

The spherical harmonics :math:`Y^l(x)` are functions defined on the sphere :math:`S^2`.

.. jupyter-execute::

    r = s2_grid()

``r`` is a grid on the sphere.

.. jupyter-execute::
    :hide-code:

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=r[..., 0].flatten(),
                y=r[..., 1].flatten(),
                z=r[..., 2].flatten(),
                mode='markers',
                marker=dict(
                    size=1,
                ),
            )
        ],
        layout=dict(
            width=500,
            height=300,
            scene=dict(
                xaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                yaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                zaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                aspectmode='manual',
                aspectratio=dict(x=3, y=3, z=3),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-5, z=5),
                    projection=dict(type='orthographic'),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0)
        )
    )
    fig.show()

Each point on the sphere has 3 component. If we plot the value of each of the 3 component separately we obtain the following figure:

.. jupyter-execute::

    plot(r)

:math:`Y^1` is the identity function. Now lets compute :math:`Y^2`, for this we take the tensor product of ``r`` with ``r`` and extract the :math:`L=2` part of it.

.. jupyter-execute::

    tp = o3.ElementwiseTensorProduct("1o", "1o", ['2e'], normalization='norm')
    y2 = tp(r, r)
    plot(y2)

:math:`Y^3` is nothing else than the :math:`L=3` part of :math:`r \otimes r \otimes r`:

.. jupyter-execute::

    tp = o3.ElementwiseTensorProduct("2e", "1o", ['3o'], normalization='norm')
    y3 = tp(y2, r)
    plot(y3)

The functions below are more efficient versions not using `ElementwiseTensorProduct`:

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    e3nn.o3.cartesian_spherical_harmonics
    e3nn.o3.angular_spherical_harmonics

.. rubric:: Details

.. automodule:: e3nn.o3.cartesian_spherical_harmonics
    :members:
    :show-inheritance:

.. automodule:: e3nn.o3.angular_spherical_harmonics
    :members:
    :show-inheritance:
