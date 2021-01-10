Model Gate Example
==================

Multipurpose equivariant neural network for point-clouds.
Made with `TensorProduct` for the linear part and `Gate` for the non linearities.

.. rubric:: Convolution

The linear part, module ``Convolution``, is inspired from the ``Depth wise Separable Convolution`` idea.
The main operation of the Convolution module is ``tp``.
It makes the atoms interact with their neighbors but does not mix the channels.
To mix the channels it is sandwich with ``lin1`` and ``lin2``.

.. literalinclude:: ../../examples/model_gate.py
    :lines: 19-64

.. rubric:: Network

The network is a simple succesion of ``Convolution`` and `Gate`.
The activation function is ReLU when dealing with even scalars and tanh of abs when dealing with even scalars.
When the parities (``p`` in `Irrep`) are provided, network is equivariant to ``O(3)``.
To relax this constraint and make it equivariant to ``SO(3)`` only, one can simply
pass all the ``irreps`` parameters to be even (``p=1`` in `Irrep`).
This is why ``irreps_sh`` is a parameter of the class ``Network``,
one can use specific ``l`` of the spherical harmonics with the correct parity ``p=(-1)^l`` (one can use `Irreps.spherical_harmonics` for that)
or consider that ``p=1`` in order to **not** be equivariant to Parity.

.. literalinclude:: ../../examples/model_gate.py
    :lines: 98-166

