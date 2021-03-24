Spherical Tensor
================

There exists 4 types of function on the sphere depending on how the parity affects it.
The representation of the coefficients are affected by this choice:

.. jupyter-execute::

    import torch
    from e3nn.io import SphericalTensor

    print(SphericalTensor(lmax=2, p_val=1, p_arg=1))
    print(SphericalTensor(lmax=2, p_val=1, p_arg=-1))
    print(SphericalTensor(lmax=2, p_val=-1, p_arg=1))
    print(SphericalTensor(lmax=2, p_val=-1, p_arg=-1))


.. jupyter-execute::

    import plotly.graph_objects as go

    def plot(traces):
        traces = [go.Surface(**d) for d in traces]
        fig = go.Figure(data=traces)
        fig.show()


In the following graph we show the four possible behavior under parity for a function on the sphere.

#. This first ball shows :math:`f(x)` unaffected by the parity
#. Then ``p_val=1`` but ``p_arg=-1`` so we see the signal flipped over the sphere but the colors are unchanged
#. For ``p_val=-1`` and ``p_arg=1`` only the value of the signal flips its sign
#. For ``p_val=-1`` and ``p_arg=-1`` both in the same time, the signal flips over the sphere and the value flip its sign

.. jupyter-execute::
    :hide-code:

    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
        nticks=3,
    )

    layout = dict(
        width=680,
        height=260,
        scene=dict(
            xaxis=dict(
                **axis,
                range=[-4, 4]
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
            aspectratio=dict(x=4, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-5, z=0),
                projection=dict(type='orthographic'),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    cmap_bwr = [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]

    def plot(traces):
        cmax = max(abs(d['surfacecolor']).max() for d in traces)
        traces = [go.Surface(**d, colorscale=cmap_bwr, cmin=-cmax, cmax=cmax) for d in traces]
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

.. jupyter-execute::

    lmax = 1
    x = torch.tensor([0.8] + [0.0, 0.0, 1.0])

    parity = -torch.eye(3)

    x = torch.stack([
        SphericalTensor(lmax, p_val, p_arg).D_from_matrix(parity) @ x
        for p_val in [+1, -1]
        for p_arg in [+1, -1]
    ])
    centers = torch.tensor([
        [-3.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

    st = SphericalTensor(lmax, 1, 1)  # p_val and p_arg set arbitrarily here
    plot(st.plotly_surface(x, centers=centers, radius=False))


.. autoclass:: e3nn.io.SphericalTensor
    :members:
    :show-inheritance:
