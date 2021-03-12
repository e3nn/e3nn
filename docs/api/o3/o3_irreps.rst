.. _Irreducible representations:

o3 - Irreps
===========

The irreducible representations, in short *irreps* (definition of irreps_) describe the action of a group on a vector space.

- Any representation can be decomposed via a change of basis into a direct sum of irreps
- Any physical quantity, under the action of :math:`O(3)`, transforms with a representation of :math:`O(3)`

The class `Irreps` represent the direct sum of irreps:

.. math::

    g \mapsto \bigoplus_{i=1}^n \sigma(g)^{p_i} D^{L_i}(g)

where :math:`(p_i = \pm 1, L_i = 0,1,2,3,\dots)_{i=1}^n` defines the `Irreps`.

- :math:`g \in O(3)`
- :math:`D^L` is the wigner_ matrix of order :math:`L`
- and :math:`\sigma(g)` is -1 if :math:`g` contains an inversion and 1 otherwise.

Irreps of :math:`O(3)` are often confused with the spherical harmonics, the relation between the irreps and the spherical harmonics is explained at :ref:`Spherical Harmonics`.

.. _irreps: https://en.wikipedia.org/wiki/Irreducible_representation
.. _wigner: https://en.wikipedia.org/wiki/Wigner_D-matrix

.. automodule:: e3nn.o3.irreps
    :members:
    :show-inheritance:
