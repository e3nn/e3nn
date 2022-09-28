.. _Spherical Harmonics:

Spherical Harmonics
===================

The spherical harmonics :math:`Y^l(x)` are functions defined on the sphere :math:`S^2`.
They form a basis of the space on function on the sphere:

.. math::

    \mathcal{F} = \{ S^2 \longrightarrow \mathbb{R} \}


On this space it is nautal how the group :math:`O(3)` acts,
Given :math:`p_a, p_v` two scalar representations:

.. math::

    [L(g) f](x) = p_v(g) f(p_a(g) R(g)^{-1} x), \quad \forall f \in \mathcal{F}, x \in S^2

:math:`L` is representation of :math:`O(3)`. But :math:`L` is not irreducible.
It can be decomposed via a change of basis into a sum of irreps,
In a handwavey notation we can write:

.. math::

    Y^T L(g) Y = 0 \oplus 1 \oplus 2 \oplus 3 \oplus \dots

where the change of basis are the spherical harmonics!
This notation is handwavey because :math:`x` is a continuous variable, and therefore the change of basis :math:`Y` is not a matrix.

As a consequence, the spherical harmonics are equivariant,

.. math::

    Y^l(R(g) x) = D^l(g) Y^l(x)

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
        betas = torch.linspace(0, math.pi, 40)
        alphas = torch.linspace(0, 2 * math.pi, 80)
        beta, alpha = torch.meshgrid(betas, alphas, indexing='ij')
        return o3.angles_to_xyz(alpha, beta)

    def trace(r, f, c, radial_abs=True):
        if radial_abs:
            a = f.abs()
        else:
            a = 1
        return dict(
            x=a * r[..., 0] + c[0],
            y=a * r[..., 1] + c[1],
            z=a * r[..., 2] + c[2],
            surfacecolor=f
        )

    def plot(data, radial_abs=True):
        r = s2_grid()
        n = data.shape[-1]
        traces = [
            trace(r, data[..., i], torch.tensor([2.0 * i - (n - 1.0), 0.0, 0.0]), radial_abs=radial_abs)
            for i in range(n)
        ]
        cmax = max(d['surfacecolor'].abs().max().item() for d in traces)
        traces = [go.Surface(**d, colorscale=cmap_bwr, cmin=-cmax, cmax=cmax) for d in traces]
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

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

Each point on the sphere has 3 components. If we plot the value of each of the 3 component separately we obtain the following figure:

.. jupyter-execute::

    plot(r, radial_abs=False)

x, y and z are represented as 3 scalar fields on 3 different spheres.
To obtain a nicer figure (that looks like the spherical harmonics shown on Wikipedia) we can deform the spheres into a shape that has its radius equal to the absolute value of the plotted quantity:

.. jupyter-execute::

    plot(r)

:math:`Y^1` is the identity function. Now let's compute :math:`Y^2`, for this we take the tensor product :math:`r \otimes r` and extract the :math:`L=2` part of it.

.. jupyter-execute::

    tp = o3.ElementwiseTensorProduct("1o", "1o", ['2e'], irrep_normalization='norm')
    y2 = tp(r, r)
    plot(y2)

Similarly, the next spherical harmonic function :math:`Y^3` is the :math:`L=3` part of :math:`r \otimes r \otimes r`:

.. jupyter-execute::

    tp = o3.ElementwiseTensorProduct("2e", "1o", ['3o'], irrep_normalization='norm')
    y3 = tp(y2, r)
    plot(y3)

The functions below are more efficient versions not using `e3nn.o3.ElementwiseTensorProduct`:

.. rubric:: Details

.. autofunction:: e3nn.o3.spherical_harmonics

.. autofunction:: e3nn.o3.spherical_harmonics_alpha_beta

.. autofunction:: e3nn.o3.Legendre
