.. _Irreducible representations:

Irreps
======

A group representation :math:`(D,V)` describe the action of a group :math:`G` on a vector space :math:`V`

.. math::

    D : G \longrightarrow \text{linear map on } V.

The irreducible representations, in short *irreps* (definition of irreps_) are the "smallest" representations.

- Any representation can be decomposed via a change of basis into a direct sum of irreps
- Any physical quantity, under the action of :math:`O(3)`, transforms with a representation of :math:`O(3)`

The irreps of :math:`SO(3)` are called the wigner_ matrices :math:`D^L`.
The irreps of the group of inversion (:math:`\{e, I\}`) are the trivial_ representation :math:`\sigma_+` and the sign representation :math:`\sigma_-`

.. math::

    \sigma_p(g) = \left \{ \begin{array}{l} 1 \text{ if } g = e \\ p \text{ if } g = I \end{array} \right..

The group :math:`O(3)` is the direct_ product of :math:`SO(3)` and inversion

.. math::

    g = r i, \quad r \in SO(3), i \in \text{inversion}.

The irreps of :math:`O(3)` are the product of the irreps of :math:`SO(3)` and inversion.
An instance of the class `Irreps` represent a direct sum of irreps of :math:`O(3)`:

.. math::

    g = r i \mapsto \bigoplus_{j=1}^n m_j \times \sigma_{p_j}(i) D^{L_j}(r)

where :math:`(m_j \in \mathbb{N}, p_j = \pm 1, L_j = 0,1,2,3,\dots)_{j=1}^n` defines the `Irreps`.

Irreps of :math:`O(3)` are often confused with the spherical harmonics, the relation between the irreps and the spherical harmonics is explained at :ref:`Spherical Harmonics`.

.. _direct: https://en.wikipedia.org/wiki/Direct_product_of_groups
.. _trivial: https://en.wikipedia.org/wiki/Trivial_representation
.. _irreps: https://en.wikipedia.org/wiki/Irreducible_representation
.. _wigner: https://en.wikipedia.org/wiki/Wigner_D-matrix

.. autoclass:: e3nn.o3.Irrep
    :members:
    :show-inheritance:

.. autoclass:: e3nn.o3.Irreps
    :members:
    :show-inheritance:
