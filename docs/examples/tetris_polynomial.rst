.. _tetris_poly:

Tetris Polynomial Example
=========================

In this example we create an *equivariant polynomial* to classify tetris.

We use the following feature of e3nn:

* `e3nn.o3.Irreps`
* `o3.spherical_harmonics`
* `e3nn.o3.FullyConnectedTensorProduct`

And the following features of `pytorch_geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_

* `radius_graph <https://github.com/rusty1s/pytorch_cluster#radius-graph>`_
* `scatter <https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html>`_

.. rubric:: the model

.. literalinclude:: ../../examples/tetris_polynomial.py
    :lines: 60-107


.. rubric:: training

.. literalinclude:: ../../examples/tetris_polynomial.py
    :lines: 111-127
    :dedent: 4

Full code `here <https://github.com/e3nn/e3nn/blob/main/examples/tetris_polynomial.py>`_
