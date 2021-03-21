.. _norm guide:

Normalization
=============

.. jupyter-execute::
    :hide-code:

    import torch

We define two kind of normalizations: ``component`` and ``norm``.

Definition
----------

component
"""""""""

``component`` normalization refers to tensors with each component of value around 1.
More precisely, the second moment of each component is 1.

.. math::

    \langle x_i^2 \rangle = 1

Examples:

* ``[1.0, -1.0, -1.0, 1.0]``
* ``[1.0, 1.0, 1.0, 1.0]`` the mean **don't** need to be zero
* ``[0.0, 2.0, 0.0, 0.0]`` this is still fine because :math:`\|x\|^2 = n`

.. jupyter-execute::

    torch.randn(10)


norm
""""

``norm`` normalization refers to tensors of norm close to 1.

.. math::

    \|x\| \approx 1

Examples:

* ``[0.5, -0.5, -0.5, 0.5]``
* ``[0.5, 0.5, 0.5, 0.5]`` the mean **don't** need to be zero
* ``[0.0, 1.0, 0.0, 0.0]``

.. jupyter-execute::

    torch.randn(10) / 10**0.5


There is just a factor :math:`\sqrt{n}` between the two normalizations.

Motivation
----------

Assuming that the weights distribution obey

.. math::

    \langle w_i \rangle = 0

    \langle w_i w_j \rangle = \sigma^2 \delta_{ij}

It imply that the two first moments of :math:`x \cdot w` (and therefore mean and variance) are only function of the second moment of :math:`x`

.. math::

    \langle x \cdot w \rangle &= \sum_i \langle x_i w_i \rangle = \sum_i \langle x_i \rangle \langle w_i \rangle = 0

    \langle (x \cdot w)^2 \rangle &= \sum_{i} \sum_{j} \langle x_i w_i x_j w_j \rangle

                                  &= \sum_{i} \sum_{j} \langle x_i x_j \rangle \langle w_i w_j \rangle

                                  &= \sigma^2 \sum_{i} \langle x_i^2 \rangle
