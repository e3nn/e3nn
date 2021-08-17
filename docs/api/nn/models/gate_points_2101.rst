Model Gate of January 2021
==========================

Multipurpose equivariant neural network for point-clouds.
Made with `e3nn.o3.TensorProduct` for the linear part and `e3nn.nn.Gate` for the nonlinearities.

.. rubric:: Convolution

The linear part, module ``Convolution``, is inspired from the ``Depth wise Separable Convolution`` idea.
The main operation of the Convolution module is ``tp``.
It makes the atoms interact with their neighbors but does not mix the channels.
To mix the channels, it is sandwiched between ``lin1`` and ``lin2``.

.. literalinclude:: ../../../../e3nn/nn/models/gate_points_2101.py
    :lines: 22-120

.. rubric:: Network

The network is a simple succession of ``Convolution`` and `e3nn.nn.Gate`.
The activation function is ReLU when dealing with even scalars and tanh of abs when dealing with even scalars.
When the parities (``p`` in `e3nn.o3.Irrep`) are provided, network is equivariant to ``O(3)``.
To relax this constraint and make it equivariant to ``SO(3)`` only, one can simply
pass all the ``irreps`` parameters to be even (``p=1`` in `e3nn.o3.Irrep`).
This is why ``irreps_sh`` is a parameter of the class ``Network``,
one can use specific ``l`` of the spherical harmonics with the correct parity ``p=(-1)^l`` (one can use `e3nn.o3.Irreps.spherical_harmonics` for that)
or consider that ``p=1`` in order to **not** be equivariant to parity.

.. literalinclude:: ../../../../e3nn/nn/models/gate_points_2101.py
    :lines: 156-336

.. automodule:: e3nn.nn.models.gate_points_2101
    :members:
    :show-inheritance:
