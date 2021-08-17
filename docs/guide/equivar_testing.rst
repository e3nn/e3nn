Equivariance Testing
====================

In `e3nn.util.test`, the library provides some tools for confirming that functions are equivariant. The main tool is `equivariance_error`, which computes the largest absolute change in output between the function applied to transformed arguments and the transform applied to the function:

.. jupyter-execute::

    import e3nn.o3
    from e3nn.util.test import equivariance_error

    tp = e3nn.o3.FullyConnectedTensorProduct("2x0e + 3x1o", "2x0e + 3x1o", "2x1o")

    equivariance_error(
        tp,
        args_in=[tp.irreps_in1.randn(1, -1), tp.irreps_in2.randn(1, -1)],
        irreps_in=[tp.irreps_in1, tp.irreps_in2],
        irreps_out=[tp.irreps_out]
    )

The keys in the output indicate the type of random transformation (``(parity, did_translation)``) and the values are the maximum componentwise error.
For convenience, the wrapper function `assert_equivariant` is provided:

.. jupyter-execute::

    from e3nn.util.test import assert_equivariant

    assert_equivariant(tp)

For typical e3nn operations `assert_equivariant` can optionally infer the input and output `e3nn.o3.Irreps`, generate random inputs when no inputs are provided, and check the error against a threshold appropriate to the current  ``torch.get_default_dtype()``.

In addition to `e3nn.o3.Irreps`-like objects, ``irreps_in`` can also contain two special values:

 * ``'cartesian_points'``: ``(N, 3)`` tensors containing XYZ points in real space that are equivariant under rotations *and* translations
 * ``None``: any input or output that is invariant and should be left alone

These can be used to test models that operate on full graphs that include position information:

.. jupyter-execute::
    :hide-code:

    kwargs = dict(
        irreps_in="3x0e + 2x1o",
        irreps_out="4x0e + 1x1o",
        max_radius=2.0,
        num_neighbors=3.0,
        num_nodes=5.0
    )

.. jupyter-execute::

    import torch
    from torch_geometric.data import Data
    from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
    from e3nn.util.test import assert_equivariant

    # kwargs = ...
    f = SimpleNetwork(**kwargs)

    def wrapper(pos, x):
        data = dict(pos=pos, x=x)
        return f(data)

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', f.irreps_in],
        irreps_out=[f.irreps_out],
    )

To test equivariance on a specific graph, ``args_in`` can be used:

.. jupyter-execute::
    :hide-code:

    my_pos = torch.randn(3, 3)
    my_x = f.irreps_in.randn(3, -1)

.. jupyter-execute::

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', f.irreps_in],
        args_in=[my_pos, my_x],
        irreps_out=[f.irreps_out],
    )

Logging
-------
``assert_equivariant`` also logs the equivariance error to the ``e3nn.util.test`` logger with level ``INFO`` regardless of whether the test fails. When running in pytest, these logs can be seen using the `"Live Logs" feature <https://docs.pytest.org/en/stable/logging.html#live-logs>`_:

.. code::

    pytest tests/ --log-cli-level info