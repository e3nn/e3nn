Equivariance Testing
====================

In `e3nn.util.test`, the library provides some tools for confirming that functions are equivariant. The main tool is `equivariance_error`, which computes the largest absolute change in output between the function applied to transformed arguments and the transform applied to the function::

    In [1]: import e3nn.o3

    In [2]: tp = e3nn.o3.FullyConnectedTensorProduct("2x0e + 3x1o", "2x0e + 3x1o", "2x1o")

    In [3]: from e3nn.util.test import equivariance_error

    In [4]: equivariance_error(
                tp,
                args_in=[tp.irreps_in1.randn(1, -1), tp.irreps_in2.randn(1, -1)],
                irreps_in=[tp.irreps_in1, tp.irreps_in2],
                irreps_out=[tp.irreps_out]
            )
    Out[4]:
    {(tensor(0.), False): tensor(1.1921e-07, grad_fn=<MaxBackward1>),
    (tensor(1.), False): tensor(1.1921e-07, grad_fn=<MaxBackward1>)}

The keys in the output indicate the type of random transformation (``(parity, did_translation)``) and the values are the maximum componentwise error.
For convenience, the wrapper function `assert_equivariant` is provided::

    In [5]: from e3nn.util.test import assert_equivariant

    In [6]: assert_equivariant(tp)

For typical e3nn operations `assert_equivariant` can optionally infer the input and output `Irreps`, generate random inputs when no inputs are provided, and check the error against a threshold appropriate to the current  ``torch.get_default_dtype()``.

In addition to `Irreps`-like objects, ``irreps_in`` can also contain two special values:

 * ``'cartesian_points'``: ``(N, 3)`` tensors containing XYZ points in real space that are equivariant under rotations *and* translations
 * ``None``: any input or output that is invariant and should be left alone

These can be used to test models that operate on full graphs that include position information::

    import torch
    from torch_geometric.data import Data
    from e3nn.nn.models.gate_points_2101 import Network
    from e3nn.util.test import assert_equivariant

    # kwargs = ...
    network = Network(**kwargs)

    def wrapper(pos, x, z):
        data = Data(pos=pos, x=x, z=z, batch=torch.zeros(pos.shape[0], dtype=torch.long))
        return f(data)

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', f.irreps_in, f.irreps_node_attr],
        irreps_out=[f.irreps_out],
    )

To test equivariance on a specific graph, ``args_in`` can be used::

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', f.irreps_in, f.irreps_node_attr],
        args_in=[my_pos, my_x, my_z],
        irreps_out=[f.irreps_out],
    )
